import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on Earh using the Haversine formula.
    Returns distance in meters
    """
    
    R = 6371000 #Earth's radius in meters
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2) ** 2
    a = np.clip(a, 0, 1)
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R*c

def get_neighbors(point_idx, points, epsilon):
    """
    Find all points within epsilon meters of points[points_idx].
    Returns list of indices.
    """
    neighbors = []
    lat1, lon1 = points[point_idx]
    
    for i in range(len(points)):
        if i == point_idx:
            continue
        lat2, lon2 = points[i]
        dist = haversine_distance(lat1, lon1, lat2, lon2)
        if dist <= epsilon:
            neighbors.append(i)
            
    return neighbors

def dbscan(points, epsilon=300, min_samples=5):
    """
    DBSCAN clustering from scratch.
    
    Args:
        points: numpy array of shape (n, 2) with [latitude, longitude]
        epsilon: maximum distance in meters between two points to be neighbors
        min_samples: minimum number of neighbors for a point to be a core point
        
    Returns:
        labels: numpy array of cluster labels (-1=noise)
    """
    n_points = len(points)
    labels = np.full(n_points, -1) #-1 means unvisited/noise
    visited = np.zeros(n_points, dtype=bool)
    cluster_id = 0
    
    for i in range(n_points):
        if visited[i]:
            continue
        
        visited[i] = True
        neighbors = get_neighbors(i, points, epsilon)
        
        if len(neighbors) < min_samples:
            # Not a core point = remains noise (label = -1) for now
            continue
        
        #Core point found - start a new cluster
        labels[i] = cluster_id
        
        #BFS expansion
        queue = list(neighbors)
        while queue:
            j = queue.pop(0)
            
            if not visited[j]:
                visited[j] = True
                j_neighbors = get_neighbors(j, points, epsilon)
                
                if len(j_neighbors) >= min_samples:
                    #j is also a core point - add its neighbors to queue
                    queue.extend(j_neighbors)
            
            #Assign to cluster if not already assigned
            if labels[j] == -1:
                labels[j] = cluster_id
        
        cluster_id +=1
        
        if cluster_id % 5 == 0:
            print(f"Found {cluster_id} clusters so far...")
    
    print(f"\n DBSCAN complete: {cluster_id} clusters, {np.sum(labels == -1)} noise points")
    return labels