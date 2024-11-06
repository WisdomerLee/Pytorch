# 해당 내용을 개발하려면 networkx라는 요소가 추가로 필요함 - pip install networkx

import networkx as nx

# graph 만들기
my_nodes = list(range(5))

H = nx.DiGraph()

# 리스트의 숫자로 node 만들기
H.add_nodes_from(my_nodes)
# 숫자로 구성된 노드들간 연결성을 아래와 같이 출발, 도착 부분으로 지정할 것
H.add_edges_from([
  (1, 0),
  (1, 2),
  (1, 3),
  (3, 2),
  (3, 4),
  (4, 0)
])
# 그래프 노드, 연결상태 그림으로 보기!
nx.draw(H, with_labels = True)

# dense adjacency matrix
nx.adjaecency_matrix(H).todense()

# 
