import networkx as nx
import matplotlib.pyplot as plt
import random
from community import community_louvain
from matplotlib.lines import Line2D

def generate_employees(num_employees):
    departments = ['GA', 'SALES', 'ENGINEERING']
    roles = ['Junior', 'Senior', 'Manager', 'Director']
    skills = ['Technical', 'Creative', 'Analytical', 'Communication', 'Leadership']
    employees = []
    for i in range(num_employees):
        employee = {
            'id': i,
            'name': f'Employee_{i}',
            'department': random.choice(departments),
            'years_experience': random.randint(1, 20),
            'role': random.choices(roles, weights=[4, 3, 2, 1])[0],  # More juniors, fewer directors
            'skills': random.sample(skills, k=random.randint(1, 3))  # Each employee has 1-3 skills
        }
        employees.append(employee)
    return employees

def create_company_graph(employees):
    G = nx.Graph()
    
    for employee in employees:
        G.add_node(employee['id'], **employee)
    
    # Create hierarchical structure
    roles = {'Director': [], 'Manager': [], 'Senior': [], 'Junior': []}
    departments = set(employee['department'] for employee in employees)
    
    for employee in employees:
        roles[employee['role']].append(employee)
    
    # Reporting relationships
    for director in roles['Director']:
        for manager in roles['Manager']:
            if director['department'] == manager['department']:
                G.add_edge(director['id'], manager['id'], relationship='reporting')
    
    for manager in roles['Manager']:
        for senior in roles['Senior']:
            if manager['department'] == senior['department'] and random.random() < 0.6:  # 60% chance of reporting to this manager
                G.add_edge(manager['id'], senior['id'], relationship='reporting')
    
    for senior in roles['Senior']:
        for junior in roles['Junior']:
            if senior['department'] == junior['department'] and random.random() < 0.4:  # 40% chance of reporting to this senior
                G.add_edge(senior['id'], junior['id'], relationship='reporting')
    
    # Project collaborations
    num_projects = len(employees) // 10  # Assume one project for every 10 employees
    for _ in range(num_projects):
        project_team = random.sample(employees, k=random.randint(3, 7))
        for i in range(len(project_team)):
            for j in range(i+1, len(project_team)):
                G.add_edge(project_team[i]['id'], project_team[j]['id'], relationship='project')
    
    # Mentorship program
    mentors = roles['Senior'] + roles['Manager'] + roles['Director']
    mentees = roles['Junior'] + roles['Senior']
    for mentee in mentees:
        if random.random() < 0.4:  # 40% chance of having a mentor
            potential_mentors = [m for m in mentors if m['department'] == mentee['department'] and m['id'] != mentee['id']]
            if potential_mentors:
                mentor = random.choice(potential_mentors)
                G.add_edge(mentor['id'], mentee['id'], relationship='mentorship')
    
    # Cross-functional teams (based on skills)
    for skill in ['Technical', 'Creative', 'Analytical', 'Communication', 'Leadership']:
        skilled_employees = [e for e in employees if skill in e['skills']]
        if len(skilled_employees) > 1:
            for i in range(len(skilled_employees)):
                for j in range(i+1, len(skilled_employees)):
                    if random.random() < 0.3:  # 30% chance of connection within skill group
                        G.add_edge(skilled_employees[i]['id'], skilled_employees[j]['id'], relationship='cross-functional')
    
    return G
def create_department_graph(employees, target_department):
    dept_employees = [e for e in employees if e['department'] == target_department]
    G = nx.Graph()
    
    for employee in dept_employees:
        G.add_node(employee['id'], **employee)
    
    # Create hierarchical structure
    roles = {'Director': [], 'Manager': [], 'Senior': [], 'Junior': []}
    for employee in dept_employees:
        roles[employee['role']].append(employee)
    
    # Reporting relationships
    for director in roles['Director']:
        for manager in roles['Manager']:
            G.add_edge(director['id'], manager['id'], relationship='reporting')
    
    for manager in roles['Manager']:
        for senior in random.sample(roles['Senior'], min(len(roles['Senior']), 3)):
            G.add_edge(manager['id'], senior['id'], relationship='reporting')
    
    for senior in roles['Senior']:
        for junior in random.sample(roles['Junior'], min(len(roles['Junior']), 5)):
            G.add_edge(senior['id'], junior['id'], relationship='reporting')
    
    # Project collaborations
    num_projects = len(dept_employees) // 5  # Assume one project for every 5 employees
    for _ in range(num_projects):
        project_team = random.sample(dept_employees, k=random.randint(3, 7))
        for i in range(len(project_team)):
            for j in range(i+1, len(project_team)):
                G.add_edge(project_team[i]['id'], project_team[j]['id'], relationship='project')
    
    # Mentorship program
    mentors = roles['Senior'] + roles['Manager'] + roles['Director']
    mentees = roles['Junior'] + roles['Senior']
    for mentee in mentees:
        if random.random() < 0.4:  # 40% chance of having a mentor
            mentor = random.choice(mentors)
            if mentor['id'] != mentee['id']:
                G.add_edge(mentor['id'], mentee['id'], relationship='mentorship')
    
    # Cross-functional teams (based on skills)
    for skill in ['Technical', 'Creative', 'Analytical', 'Communication', 'Leadership']:
        skilled_employees = [e for e in dept_employees if skill in e['skills']]
        if len(skilled_employees) > 1:
            for i in range(len(skilled_employees)):
                for j in range(i+1, len(skilled_employees)):
                    if random.random() < 0.3:  # 30% chance of connection within skill group
                        G.add_edge(skilled_employees[i]['id'], skilled_employees[j]['id'], relationship='cross-functional')
    
    return G

def analyze_and_visualize_department_network(G, department):
    if len(G) == 0:
        print(f"No employees found in the {department} department.")
        return

    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    
    central_nodes = set(sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:3])
    knowledge_brokers = set(sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)[:3])
    
    partition = community_louvain.best_partition(G)
    
    num_clusters = len(set(partition.values()))
    color_map = plt.cm.get_cmap('tab20')
    node_colors = [color_map(partition[node] / num_clusters) for node in G.nodes()]
    
    node_sizes = []
    node_shapes = []
    for node in G.nodes():
        size = 300 if node in central_nodes or node in knowledge_brokers else 100
        node_sizes.append(size)
        if node in central_nodes and node in knowledge_brokers:
            node_shapes.append('s')  # square for both
        elif node in central_nodes:
            node_shapes.append('^')  # triangle for central nodes
        elif node in knowledge_brokers:
            node_shapes.append('D')  # diamond for knowledge brokers
        else:
            node_shapes.append('o')  # circle for regular nodes
    
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    plt.figure(figsize=(15, 10))
    
    # Draw nodes
    for shape in set(node_shapes):
        node_list = [node for node, node_shape in zip(G.nodes(), node_shapes) if node_shape == shape]
        nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_color=[node_colors[list(G.nodes()).index(n)] for n in node_list],
                               node_size=[node_sizes[list(G.nodes()).index(n)] for n in node_list], node_shape=shape, alpha=0.8)
    
    # Draw edges with different colors for different relationships
    edge_colors = {'reporting': 'gray', 'project': 'blue', 'mentorship': 'green', 'cross-functional': 'red'}
    for relationship, color in edge_colors.items():
        edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['relationship'] == relationship]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color, alpha=0.5)
    
    labels = {node: f"{G.nodes[node]['name']}\n({G.nodes[node]['role']})" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title(f"{department} Department Network Analysis")
    
    # Create legend for nodes and edges
    node_legend = [
        Line2D([0], [0], marker='o', color='w', label='Regular Employee', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='^', color='w', label='Central Node', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='D', color='w', label='Knowledge Broker', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='s', color='w', label='Central & Broker', markerfacecolor='gray', markersize=10)
    ]
    edge_legend = [Line2D([0], [0], color=color, label=relationship) for relationship, color in edge_colors.items()]
    plt.legend(handles=node_legend + edge_legend, loc='upper right', bbox_to_anchor=(1.3, 1))
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"\n{department} Department Analysis:")
    print("\nCentral Nodes (High Degree Centrality):")
    for node in central_nodes:
        print(f"{G.nodes[node]['name']} - Role: {G.nodes[node]['role']} - Years: {G.nodes[node]['years_experience']}")
    
    print("\nKnowledge Brokers (High Betweenness Centrality):")
    for node in knowledge_brokers:
        print(f"{G.nodes[node]['name']} - Role: {G.nodes[node]['role']} - Years: {G.nodes[node]['years_experience']}")
    
    print("\nCluster Analysis:")
    cluster_sizes = {}
    cluster_roles = {}
    for node, cluster in partition.items():
        cluster_sizes[cluster] = cluster_sizes.get(cluster, 0) + 1
        role = G.nodes[node]['role']
        if cluster not in cluster_roles:
            cluster_roles[cluster] = {}
        cluster_roles[cluster][role] = cluster_roles[cluster].get(role, 0) + 1
    
    for cluster, size in sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True):
        print(f"\nCluster {cluster} - Size: {size}")
        print("Role Distribution:")
        for role, count in sorted(cluster_roles[cluster].items(), key=lambda x: x[1], reverse=True):
            print(f"  {role}: {count} ({count/size*100:.1f}%)")

    # Relationship type analysis
    relationship_counts = {rel: 0 for rel in edge_colors.keys()}
    for _, _, data in G.edges(data=True):
        relationship_counts[data['relationship']] += 1
    
    print("\nRelationship Type Analysis:")
    for rel, count in relationship_counts.items():
        print(f"{rel.capitalize()}: {count}")
def analyze_and_visualize_cross_functional_network(G, department):
    if len(G) == 0:
        print(f"No employees found in the {department} department.")
        return

    # Create a subgraph with only cross-functional edges
    cross_functional_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['relationship'] == 'cross-functional']
    G_cross = G.edge_subgraph(cross_functional_edges)

    if len(G_cross) == 0:
        print(f"No cross-functional relationships found in the {department} department.")
        return

    degree_centrality = nx.degree_centrality(G_cross)
    betweenness_centrality = nx.betweenness_centrality(G_cross)
    
    central_nodes = set(sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:3])
    knowledge_brokers = set(sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)[:3])
    
    partition = community_louvain.best_partition(G_cross)
    
    num_clusters = len(set(partition.values()))
    color_map = plt.cm.get_cmap('tab20')
    node_colors = [color_map(partition[node] / num_clusters) for node in G_cross.nodes()]
    
    node_sizes = []
    node_shapes = []
    for node in G_cross.nodes():
        size = 300 if node in central_nodes or node in knowledge_brokers else 100
        node_sizes.append(size)
        if node in central_nodes and node in knowledge_brokers:
            node_shapes.append('s')  # square for both
        elif node in central_nodes:
            node_shapes.append('^')  # triangle for central nodes
        elif node in knowledge_brokers:
            node_shapes.append('D')  # diamond for knowledge brokers
        else:
            node_shapes.append('o')  # circle for regular nodes
    
    pos = nx.spring_layout(G_cross, k=0.5, iterations=50)
    plt.figure(figsize=(15, 10))
    
    # Draw nodes
    for shape in set(node_shapes):
        node_list = [node for node, node_shape in zip(G_cross.nodes(), node_shapes) if node_shape == shape]
        nx.draw_networkx_nodes(G_cross, pos, nodelist=node_list, node_color=[node_colors[list(G_cross.nodes()).index(n)] for n in node_list],
                               node_size=[node_sizes[list(G_cross.nodes()).index(n)] for n in node_list], node_shape=shape, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G_cross, pos, edge_color='red', alpha=0.5)
    
    labels = {node: f"{G.nodes[node]['name']}\n({G.nodes[node]['role']})" for node in G_cross.nodes()}
    nx.draw_networkx_labels(G_cross, pos, labels, font_size=8)
    
    plt.title(f"{department} Department Cross-Functional Network Analysis")
    
    # Create legend for nodes
    node_legend = [
        Line2D([0], [0], marker='o', color='w', label='Regular Employee', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='^', color='w', label='Central Node', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='D', color='w', label='Knowledge Broker', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='s', color='w', label='Central & Broker', markerfacecolor='gray', markersize=10)
    ]
    plt.legend(handles=node_legend, loc='upper right', bbox_to_anchor=(1.3, 1))
    
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=False)
    
    print(f"\n{department} Department Cross-Functional Network Analysis:")
    print("\nCentral Nodes (High Degree Centrality):")
    for node in central_nodes:
        print(f"{G.nodes[node]['name']} - Role: {G.nodes[node]['role']} - Years: {G.nodes[node]['years_experience']} - Skills: {', '.join(G.nodes[node]['skills'])}")
    
    print("\nKnowledge Brokers (High Betweenness Centrality):")
    for node in knowledge_brokers:
        print(f"{G.nodes[node]['name']} - Role: {G.nodes[node]['role']} - Years: {G.nodes[node]['years_experience']} - Skills: {', '.join(G.nodes[node]['skills'])}")
    
    print("\nCluster Analysis:")
    cluster_sizes = {}
    cluster_roles = {}
    cluster_skills = {}
    for node, cluster in partition.items():
        cluster_sizes[cluster] = cluster_sizes.get(cluster, 0) + 1
        role = G.nodes[node]['role']
        skills = G.nodes[node]['skills']
        if cluster not in cluster_roles:
            cluster_roles[cluster] = {}
            cluster_skills[cluster] = {}
        cluster_roles[cluster][role] = cluster_roles[cluster].get(role, 0) + 1
        for skill in skills:
            cluster_skills[cluster][skill] = cluster_skills[cluster].get(skill, 0) + 1
    
    for cluster, size in sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True):
        print(f"\nCluster {cluster} - Size: {size}")
        print("Role Distribution:")
        for role, count in sorted(cluster_roles[cluster].items(), key=lambda x: x[1], reverse=True):
            print(f"  {role}: {count} ({count/size*100:.1f}%)")
        print("Skill Distribution:")
        for skill, count in sorted(cluster_skills[cluster].items(), key=lambda x: x[1], reverse=True):
            print(f"  {skill}: {count} ({count/size*100:.1f}%)")

    print("\nNetwork Statistics:")
    print(f"Number of nodes: {G_cross.number_of_nodes()}")
    print(f"Number of edges: {G_cross.number_of_edges()}")
    print(f"Average clustering coefficient: {nx.average_clustering(G_cross):.3f}")
    print(f"Network density: {nx.density(G_cross):.3f}")

# Main execution
# num_employees = 200
# employees = generate_employees(num_employees)

# # Select a department for analysis
# target_department = 'ENGINEERING'  # You can change this to any department

# G = create_department_graph(employees, target_department)
# analyze_and_visualize_cross_functional_network(G, target_department)
# # Main execution
# #num_employees = 200
# #employees = generate_employees(num_employees)

# # Select a department for analysis
# #target_department = 'ENGINEERING'  # You can change this to any department

# #G = create_department_graph(employees, target_department)
# analyze_and_visualize_department_network(G, target_department)

num_employees = 200
employees = generate_employees(num_employees)

# Create graph for all departments
G = create_company_graph(employees)

# Analyze and visualize the entire company network
analyze_and_visualize_department_network(G, "All Departments")

# If you want to analyze cross-functional relationships across the entire company
analyze_and_visualize_cross_functional_network(G, "All Departments")

# If you still want to analyze individual departments, you can create subgraphs
for department in set(employee['department'] for employee in employees):
    department_employees = [e['id'] for e in employees if e['department'] == department]
    G_dept = G.subgraph(department_employees)
    analyze_and_visualize_department_network(G_dept, department)
    analyze_and_visualize_cross_functional_network(G_dept, department)