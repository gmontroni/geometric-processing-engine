import numpy as np
from vet.vector_operators import normalize, norm3d

class VET:
    
    def __init__(self, V, T):
        
        # Inicializa a estrutura
        self.V = np.zeros((0,3), dtype=float)
        self.E = []
        self.T = np.zeros((0,3), dtype=int)

        # Insere os vertices
        self.add_vert(V)
        self.nv = self.V.shape[0] # ATENCAO: lembrar de atualizar

        # Insere as faces
        self.add_tri(T)
        self.nt = self.T.shape[0] # ATENCAO: lembrar de atualizar
        
    def info(self):
        print('Vertices:', self.nv, '--- Faces:', self.nt)

    def add_vert(self, V):
        if self.V.shape[0] == 0:
            self.V = V
            self.E = [[]] * V.shape[0]
        else:
            self.V = np.vstack((self.V, V))
            self.E.append([[] * V.shape[0]])

    def add_tri(self, T):

        # O indice dos triangulos inseridos comecam a partir de size(T,1)+1.

        nt_old = self.T.shape[0]
        nt_to_add = T.shape[0]

        # Adicionando o triangulo na lista
        if self.T.shape[0] == 0:
            self.T = T
        else:
            self.T = np.vstack((self.T, T))

        # Para cada aresta do novo triangulo, encontra o triangulo adjacente
        for n in range(nt_to_add):
            idx = nt_old + n

            # adiciona este triangulo a lista de estrela dos vertices
            self.E[T[n,0]] = np.unique(self.E[T[n,0]] + [idx]).tolist()
            self.E[T[n,1]] = np.unique(self.E[T[n,1]] + [idx]).tolist()
            self.E[T[n,2]] = np.unique(self.E[T[n,2]] + [idx]).tolist()
    
    def compute_area(self):
        area = 0.0
        for t in self.T:
            a, b, c = t
            u = self.V[b, :] - self.V[a, :]
            v = self.V[c, :] - self.V[a, :]
            area += 0.5 * np.linalg.norm(np.cross(u,v))
        return area
    
    def compute_ring(self, vidx):
        S = self.E[vidx]
        if not S:
            ring = []
            return ring

        ring = []
        for s in S:
            ring.append(self.T[s, :])

        ring = np.unique(ring)
        ring = ring[ring != vidx]
    
        return ring.tolist()

    def compute_k_ring(self, vidx, k):
        
        old_ring = [vidx]
        new_ring = []
        
        for i in range(k):
            for v in old_ring:
                ring = self.compute_ring(v)
                new_ring = np.unique(new_ring + ring).tolist()
                
            old_ring = new_ring
        return new_ring
    
    def computeFacesNormals(self):
        normals = np.zeros((self.nt,3))
        for i in range(self.nt):
            p1 = self.V[self.T[i,0],:]
            p2 = self.V[self.T[i,1],:]
            p3 = self.V[self.T[i,2],:]
            u = p2 - p1
            v = p3 - p1
            n = np.cross(u,v)
            normals[i,:] = n # Sem normalizar - info da area
            # normals[i,:] = n/np.linalg.norm(n)

        return normals

    def computeVerticesNormals(self):
        F = self.computeFacesNormals()
        Vnormals = np.zeros(self.V.shape)
        for i in range(self.nv):
            normals = F[self.E[i],:]
            vnormal = np.sum(normals, axis=0).reshape((1,3))
            Vnormals[i,:] = vnormal / np.linalg.norm(vnormal)
        return Vnormals

    def computeOrthonormalBase(self):
        # Sistema de coordenadas local (e,f,n)
        # Tnormals ==> e; Bnormals ==> f; Vnormals ==> n
        # OBS: não é contínuo :(

        Vnormals = self.computeVerticesNormals()
        Bnormals = np.zeros(self.V.shape)
        Tnormals = np.zeros(self.V.shape)
        for i in range(self.V.shape[0]):
            x, y, z = Vnormals[i,:]
            temp = np.array([[-y,  x, 0],
                             [-z,  0, x],
                             [ 0, -z, y]])
            mags = np.array([y*y + x*x, z*z + x*x, z*z + y*y])
            idx = mags.argmax()
            Bnormals[i,:] = temp[idx,:]
            Tnormals[i,:] = np.cross(temp[idx,:], Vnormals[i,:])

        Bnormals = normalize(Bnormals)
        Tnormals = normalize(Tnormals)

        return Tnormals, Bnormals, Vnormals

    def compute_min_edge_length(self):
        lengths = []
        for t in self.T:
            v0, v1, v2 = t[:]
            e0 = self.V[v1] - self.V[v0]
            e1 = self.V[v2] - self.V[v0]
            e2 = self.V[v2] - self.V[v1]
            lengths.append(norm3d(e0))
            lengths.append(norm3d(e1))
            lengths.append(norm3d(e2))
        return np.min(lengths)
                
    def compute_boundary(self):
        tri_boundary = []
        pts_boundary = set()

        for t in range(self.nt):
            v0, v1, v2 = self.T[t,:]
            
            #v0, v1
            inter = np.intersect1d(self.E[v0], self.E[v1])
            if len(inter) == 1:
                pts_boundary.add(v0)
                pts_boundary.add(v1)
                tri_boundary.append(t)
                continue

            #v1, v2
            inter = np.intersect1d(self.E[v1], self.E[v2])
            if len(inter) == 1:
                pts_boundary.add(v1)
                pts_boundary.add(v2)
                tri_boundary.append(t)
                continue

            #v2, v0
            inter = np.intersect1d(self.E[v2], self.E[v0])
            if len(inter) == 1:
                pts_boundary.add(v2)
                pts_boundary.add(v0)
                tri_boundary.append(t)
                continue

        return list(pts_boundary), tri_boundary