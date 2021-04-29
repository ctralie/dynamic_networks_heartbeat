import matplotlib.pyplot as plt
import seaborn as sns
#from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import skimage.io
from scipy import ndimage
import numpy as np
import numpy.linalg as linalg
import time
import os
import robust_laplacian
import polyscope as ps
import scipy.sparse.linalg as sla
from scipy import sparse
from persim import plot_diagrams
from sklearn.decomposition import PCA
import deepdish as dd
from scipy.spatial import KDTree

def imreadf(filename):
    #Read in file, converting image byte array to little endian float
    I = skimage.io.imread(filename)
    #Image is stored in BGRA format so convert to RGBA
    I = I[:, :, [2, 1, 0, 3]]
    shape = I.shape
    I = I.flatten()
    IA = bytearray(I.tolist())
    I = np.frombuffer(IA, dtype=np.dtype('<f4'))
    return np.reshape(I, shape[0:2])  

def imwritef(I, filename):
    IA = I.flatten().tolist()
    IA = struct.pack("%if"%len(IA), *IA)
    IA = np.fromstring(IA, dtype=np.uint8)
    IA = IA.reshape([I.shape[0], I.shape[1], 4]) ##Tricky!!  Numpy is "low-order major" and the order I have things in is 4bytes per pixel, then columns, then rows.  These are specified in reverse order
    print("IA.shape = ", IA.shape)
    #Convert from RGBA format to BGRA format like the real sense saver did
    IA = IA[:, :, [2, 1, 0, 3]]
    skimage.io.imsave(filename, IA)

#Use the uv coorinates to map into the array of colors "C"
#using bilinear interpolation.  Out of bounds values are by default gray
def getColorsFromMap(u, v, C, mask):
    #The uv coordinates are in [0, 1]x[0, 1] so scale to [0, height]x[0, width]
    [H, W] = [C.shape[0], C.shape[1]]
    thisu = u[mask > 0]*W
    thisv = v[mask > 0]*H
    #Round out of bounds indices to the edge for now
    thisu[thisu >= W] = W-1
    thisu[thisu < 0] = 0
    thisv[thisv >= H] = H-1
    thisv[thisv < 0] = 0
    N = len(thisu)
    #Do bilinear interpolation on grid
    m = mask.flatten()
    CAll = np.reshape(C, [C.shape[0]*C.shape[1], C.shape[2]])
    c = CAll[m > 0, :]
    
    #utop, ubottom
    ul = np.array(np.floor(thisu), dtype=np.int64)
    ur = ul + 1
    ur[ur >= W] = W-1
    #vteft, vbight
    vt = np.array(np.floor(thisv), dtype=np.int64)
    vb = vt + 1
    vb[vb >= H] = H-1
    c = (ur-thisu)[:, None]*(CAll[vt*W+ul, :]*(vb-thisv)[:, None] + CAll[vb*W+ul, :]*(thisv-vt)[:, None]) + (thisu-ul)[:, None]*(CAll[vt*W+ur, :]*(vb-thisv)[:, None] + CAll[vb*W+ur, :]*(thisv-vt)[:, None])
    #Set out of bounds pixels to gray (since color/depth aren't aligned perfectly)
    thisu = u[mask > 0]*W
    thisv = v[mask > 0]*H
    c[thisu >= W, :] = np.array([0.5, 0.5, 0.5])
    c[thisu < 0, :] = np.array([0.5, 0.5, 0.5])
    c[thisv >= H, :] = np.array([0.5, 0.5, 0.5])
    c[thisv < 0, :] = np.array([0.5, 0.5, 0.5])
    return c
    

def getFrame(foldername, index, loadColor = False, plotFrame = False):
    depthFile = "%s/B-depth-float%i.png"%(foldername, index)
    xyFile = "%s/B-cloud%i.png"%(foldername, index)
    Z = imreadf(depthFile)
    XYZ = imreadf(xyFile)
    X = XYZ[:, 0:-1:3]
    Y = XYZ[:, 1:-1:3]
    uvname = "%s/B-depth-uv%i.png"%(foldername, index)
    u = np.zeros(Z.shape)
    v = np.zeros(Z.shape)
    C = 0.5*np.ones((Z.shape[0], Z.shape[1], 3)) #Default gray
    loadedColor = False
    if loadColor and os.path.exists(uvname):
        uv = imreadf(uvname)
        u = uv[:, 0::2]
        v = uv[:, 1::2]
        C = skimage.io.imread("%s/B-color%i.png"%(foldername, index)) / 255.0
        loadedColor = True
    if plotFrame:
        x = X[Z > 0]
        y = Y[Z > 0]
        z = Z[Z > 0]
        c = getColorsFromMap(u, v, C, Z > 0)
        fig = plt.figure()
        #ax = Axes3D(fig)
        plt.scatter(x, y, 30, c)
        plt.show()
    return [X, Y, Z, C, u, v, loadedColor]

class RealSenseVideo(object):
    def __init__(self):
        self.Xs = np.zeros((0, 0, 0))
        self.Ys = np.zeros((0, 0, 0))
        self.Zs = np.zeros((0, 0, 0))
        self.Cs = np.zeros((0, 0, 0, 0)) #Color frames
        self.us = np.zeros((0, 0, 0)) #depth to color map horiz coordinate
        self.vs = np.zeros((0, 0, 0)) #depth to color map vert coordinate
        self.loadedColor = False

    def load_video(self, foldername, NFrames, loadColor = True):
        if NFrames <= 0:
            return
        shape = getFrame(foldername, 0)[0].shape
        Xs = np.zeros((shape[0], shape[1], NFrames))
        Ys = np.zeros(Xs.shape)
        Zs = np.zeros(Xs.shape)
        Cs = 0.5*np.ones([Xs.shape[0], Xs.shape[1], 3, NFrames])
        us = np.zeros(Xs.shape)
        vs = np.zeros(Xs.shape)
        for i in range(NFrames):
            print("Loading %s frame %i"%(foldername, i))
            [Xs[:, :, i], Ys[:, :, i], Zs[:, :, i], C, us[:, :, i], vs[:, :, i], self.loadedColor] = getFrame(foldername, i, loadColor)
            if self.loadedColor:
                if i == 0:
                    Cs = np.zeros((C.shape[0], C.shape[1], C.shape[2], NFrames))
                Cs[:, :, :, i] = C
        self.Xs = Xs
        self.Ys = Ys
        self.Zs = Zs
        self.Cs = Cs
        self.us = us
        self.vs = vs

    def get_mask(self):
        #Narrow down to pixels which measured something every frame
        counts = np.sum(self.Zs > 0, 2)
        Mask = (counts == self.Zs.shape[2])
        #Find biggest connected component out of remaining pixels
        ILabel, NLabels = ndimage.label(Mask)
        idx = np.argmax(ndimage.sum(Mask, ILabel, range(NLabels+1)))
        Mask = (ILabel == idx)
        return np.array(Mask, dtype=int)
    
    def make_mesh_frame(self, i, Mask = np.array([])):
        if self.Xs.shape[2] == 0:
            return
        X = self.Xs[:, :, i]
        Y = self.Ys[:, :, i]
        Z = self.Zs[:, :, i]
        #Come up with vertex indices in the mask
        if Mask.size == 0:
            # Pick largest connected component
            Mask = np.array(Z > 0, dtype=int)
            ILabel, NLabels = ndimage.label(Mask)
            idx = np.argmax(ndimage.sum(Mask, ILabel, range(NLabels+1)))
            Mask = np.array(ILabel == idx, dtype=int)
        nV = np.sum(Mask)
        Mask[Mask > 0] = np.arange(nV) + 1
        Mask = Mask - 1
        VPos = np.zeros((nV, 3))
        VPos[:, 0] = X[Mask >= 0]
        VPos[:, 1] = Y[Mask >= 0]
        VPos[:, 2] = -Z[Mask >= 0]
        #Add lower right triangle
        v1 = Mask[0:-1, 0:-1].flatten()
        v2 = Mask[1:, 0:-1].flatten()
        v3 = Mask[1:, 1:].flatten()
        N = v1.size
        ITris1 = np.concatenate((np.reshape(v1, [N, 1]), np.reshape(v2, [N, 1]), np.reshape(v3, [N, 1])), 1)
        #Add upper right triangle
        v1 = Mask[0:-1, 0:-1].flatten()
        v2 = Mask[1:, 1:].flatten()
        v3 = Mask[0:-1, 1:].flatten()
        N = v1.size
        ITris2 = np.concatenate((np.reshape(v1, [N, 1]), np.reshape(v2, [N, 1]), np.reshape(v3, [N, 1])), 1)
        ITris = np.concatenate((ITris1, ITris2), 0)
        #Only retain triangles which have all three points
        ITris = ITris[np.sum(ITris == -1, 1) == 0, :]
        #Only retain vertices which are adjacent to at least one triangle
        Vs = np.unique(ITris.flatten())
        included = np.zeros(VPos.shape[0])
        included[Vs] = 1
        reindex = np.zeros(VPos.shape[0])
        reindex[included > 0] = np.arange(np.sum(included))
        VPos = VPos[included > 0, :]
        ITris = reindex[ITris]
        return VPos, ITris

def get_edges(VPos, ITris):
    """
    Given a list of triangles, return an array representing the edges
    Parameters
    ----------
    VPos : ndarray (N, 3)
        Array of points in 3D
    ITris : ndarray (M, 3)
        Array of triangles connecting points, pointing to vertex indices
    Returns: I, J
        Two parallel 1D arrays with indices of edges
    """
    N = VPos.shape[0]
    M = ITris.shape[0]
    I = np.zeros(M*6)
    J = np.zeros(M*6)
    V = np.ones(M*6)
    for shift in range(3): 
        #For all 3 shifts of the roles of triangle vertices
        #to compute different cotangent weights
        [i, j, k] = [shift, (shift+1)%3, (shift+2)%3]
        I[shift*M*2:shift*M*2+M] = ITris[:, i]
        J[shift*M*2:shift*M*2+M] = ITris[:, j] 
        I[shift*M*2+M:shift*M*2+2*M] = ITris[:, j]
        J[shift*M*2+M:shift*M*2+2*M] = ITris[:, i] 
    L = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    return L

def get_curvature(VPos, ITris, n_neighbs=50):
    # Approximation and eigen decomposition of the shape operator
    # https://github.com/alecjacobson/geometry-processing-curvature
    N = VPos.shape[0]
    """
    L = get_edges(VPos, ITris)
    # Get 4-ring neighbors of each vertex
    L2 = L.dot(L)
    L3 = L2.dot(L)
    L4 = L3.dot(L)
    L = L + L2 + L3 + L4
    """
    tree = KDTree(VPos)
    neighbs = tree.query(VPos, n_neighbs)[1]

    curv = np.zeros((N, 2))
    pca = PCA(n_components=3)
    
    for i in range(N):
        try:
            J = neighbs[i, :]
            VJ = VPos[J, :]
            VJ -= np.mean(VJ, 0)
            Y = pca.fit_transform(VJ)
            u, v, w = Y[:, 0], Y[:, 1], Y[:, 2]
            A = np.array([u, v, u*u, u*v, v*v]).T
            AInv = linalg.pinv(A)
            [a1, a2, a3, a4, a5] = AInv.dot(w)
            E = 1+a1**2
            F = a1*a2
            G = 1+a2**2
            e = 2*a3/np.sqrt(a1**1+1+a2**2)
            f = a4/np.sqrt(a1**2+1+a2**2)
            g = 2*a5/np.sqrt(a1**2+1+a2**2)

            S1 = np.array([[e, f], [f, g]])
            S2 = np.linalg.inv(np.array([[E, F], [F, G]]))
            S = -S1.dot(S2)
            lams = linalg.eig(S)[0]
            curv[i, :] = lams 
        except e:
            print(e)
    
    return curv
        

def do0DSublevelsetFiltrationMesh(VPos, ITris, x):
    from ripser import ripser
    N = VPos.shape[0]
    # Add edges between adjacent points in the mesh    
    I, J = get_edges(VPos, ITris).nonzero()
    V = np.maximum(x[I], x[J])
    # Add vertex birth times along the diagonal of the distance matrix
    I = np.concatenate((I, np.arange(N)))
    J = np.concatenate((J, np.arange(N)))
    V = np.concatenate((V, x))
    #Create the sparse distance matrix
    D = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    return ripser(D, distance_matrix=True, maxdim=0)['dgms'][0]




#Comparing real sense to kinect frames
if __name__ == '__main__':
    N = 200 # Number of frames to load

    v = RealSenseVideo()
    v.load_video("Chris_Neck_Color_F200", N)
    
    meshes = []
    ps.init()
    allevals = np.array([])
    # First autotune heat time by looking at all eigenvalues
    min_curv = np.inf
    max_curv = -np.inf
    for i in range(N):
        print("Processing frame {}".format(i))
        verts, faces = v.make_mesh_frame(i)
        curv = get_curvature(verts, faces)
        min_curv = min(min_curv, np.min(curv))
        max_curv = max(max_curv, np.max(curv))
        meshes.append({'verts':verts, 'faces':faces, 'curv':curv})
    bins = np.linspace(min_curv, max_curv, 50)

    # Finally, output result
    plt.figure(figsize=(12, 6))
    all_dgms = []
    all_hists = []
    min1 = np.inf
    min2 = np.inf
    max1 = -np.inf
    max2 = -np.inf
    for i in range(N):
        verts, faces, curv = meshes[i]['verts'], meshes[i]['faces'], meshes[i]['curv']
        curv = curv[:, 0]*curv[:, 1]
        hist = np.histogram(curv, bins)[0]
        all_hists.append(hist)
        dd.io.save("all_hists.h5", {"all_hists":all_hists, "bins":bins})
        verts[:, [0, 2]] *= -1 # Flip around
        #ps.register_surface_mesh("Mesh", verts, faces, smooth_shade=True)
        ps_cloud = ps.register_point_cloud("curv", verts)
        ps_cloud.add_scalar_quantity("curv", curv, enabled=True)
        ps.screenshot("{}.png".format(i))
        ps.remove_point_cloud("curv")
        dgmup = do0DSublevelsetFiltrationMesh(verts, faces, curv)
        dgmup = dgmup[np.isfinite(dgmup[:, 1]), :]
        if dgmup.size > 0:
            min1 = min(np.min(dgmup), min1)
            max1 = max(np.max(dgmup), max1)

        dgmdown = do0DSublevelsetFiltrationMesh(verts, faces, -curv)
        dgmdown = dgmdown[np.isfinite(dgmdown[:, 1]), :]
        if dgmdown.size > 0:
            min2 = min(np.min(dgmdown), min2)
            max2 = max(np.max(dgmdown), max2)

        all_dgms.append({'dgmup':dgmup, 'dgmdown':dgmdown})
        dd.io.save("all_dgms.h5", {'all_dgms':all_dgms})
    
    range1 = max1 - min1
    min1 -= 0.1*range1
    max1 += 0.1*range1
    range2 = max2 - min2
    min2 -= 0.1*range2
    max2 += 0.1*range2

    plt.figure(figsize=(12, 6))
    for i in range(N):
        dgmup, dgmdown = all_dgms[i]['dgmup'], all_dgms[i]['dgmdown']
        plt.clf()
        plt.subplot(121)
        if dgmup.size > 0:
            plot_diagrams(dgmup)
            plt.xlim([min1, max1])
            plt.ylim([min1, max1])
        plt.subplot(122)
        if dgmdown.size > 0:
            plot_diagrams(dgmdown)
            plt.xlim([min2, max2])
            plt.ylim([min2, max2])
        plt.savefig("DGM{}.png".format(i))

