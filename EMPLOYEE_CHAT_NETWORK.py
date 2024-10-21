import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import datetime

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

def generate_chat_interactions(employees, num_days=30, avg_chats_per_day=5):
    interactions = []
    start_date = datetime.date.today() - datetime.timedelta(days=num_days)
    
    for day in range(num_days):
        current_date = start_date + datetime.timedelta(days=day)
        num_chats = random.randint(avg_chats_per_day - 2, avg_chats_per_day + 2)
        
        for _ in range(num_chats):
            sender = random.choice(employees)
            receiver = random.choice(employees)
            while receiver == sender:
                receiver = random.choice(employees)
            
            interaction = {
                'date': current_date,
                'sender_id': sender['id'],
                'receiver_id': receiver['id'],
                'message_count': random.randint(1, 20)  # Number of messages in this chat
            }
            interactions.append(interaction)
    
    return interactions

def create_chat_network(employees, interactions):
    G = nx.Graph()
    
    # Add nodes
    for employee in employees:
        G.add_node(employee['id'], **employee)
    
    # Add edges based on chat interactions
    edge_weights = defaultdict(int)
    for interaction in interactions:
        sender = interaction['sender_id']
        receiver = interaction['receiver_id']
        weight = interaction['message_count']
        edge_weights[(sender, receiver)] += weight
    
    for (sender, receiver), weight in edge_weights.items():
        G.add_edge(sender, receiver, weight=weight)
    
    return G

def plot_chat_network(G, title="Employee Chat Network"):
    plt.figure(figsize=(20, 15))
    
    # Calculate node sizes based on degree centrality
    degree_centrality = nx.degree_centrality(G)
    node_sizes = [v * 5000 for v in degree_centrality.values()]
    
    # Calculate edge weights
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights)
    normalized_weights = [w / max_weight for w in edge_weights]
    
    # Set up colors for departments
    departments = set(nx.get_node_attributes(G, 'department').values())
    color_map = plt.cm.get_cmap('tab10')
    color_dict = {dept: color_map(i/len(departments)) for i, dept in enumerate(departments)}
    node_colors = [color_dict[G.nodes[node]['department']] for node in G.nodes()]
    
    # Position nodes using force-directed layout
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=normalized_weights, alpha=0.3, edge_color='gray')
    
    # Add labels
    labels = {node: f"{G.nodes[node]['name']}\n({G.nodes[node]['role']})" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # Create legend for departments
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=dept,
                                  markerfacecolor=color, markersize=10)
                       for dept, color in color_dict.items()]
    plt.legend(handles=legend_elements, title="Departments", loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def analyze_chat_network(G):
    print("\nChat Network Analysis:")
    print(f"Number of employees: {G.number_of_nodes()}")
    print(f"Number of chat connections: {G.number_of_edges()}")
    
    # Degree analysis
    degrees = [d for n, d in G.degree()]
    avg_degree = sum(degrees) / len(degrees)
    print(f"\nAverage number of chat connections per employee: {avg_degree:.2f}")
    
    # Most central employees (by degree centrality)
    degree_centrality = nx.degree_centrality(G)
    top_central = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 most central employees (by number of chat connections):")
    for node_id, centrality in top_central:
        employee = G.nodes[node_id]
        print(f"  {employee['name']} ({employee['role']}, {employee['department']}): {centrality:.3f}")
    
    # Most active chat connections
    top_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:5]
    print("\nTop 5 most active chat connections:")
    for u, v, data in top_edges:
        employee1 = G.nodes[u]
        employee2 = G.nodes[v]
        print(f"  {employee1['name']} <-> {employee2['name']}: {data['weight']} messages")
    
    # Department interaction analysis
    dept_interactions = defaultdict(int)
    for u, v, data in G.edges(data=True):
        dept1 = G.nodes[u]['department']
        dept2 = G.nodes[v]['department']
        if dept1 != dept2:
            dept_interactions[tuple(sorted((dept1, dept2)))] += data['weight']
    
    print("\nTop interdepartmental interactions:")
    top_dept_interactions = sorted(dept_interactions.items(), key=lambda x: x[1], reverse=True)[:5]
    for (dept1, dept2), weight in top_dept_interactions:
        print(f"  {dept1} <-> {dept2}: {weight} messages")

# Main execution
num_employees = 200
employees = generate_employees(num_employees)
chat_interactions = generate_chat_interactions(employees, num_days=30, avg_chats_per_day=5)

G_chat = create_chat_network(employees, chat_interactions)
plot_chat_network(G_chat)
analyze_chat_network(G_chat)