import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from scipy.stats import pearsonr

# Set random seed for reproducibility
np.random.seed(42)

# Generate correlated data similar to MATLAB code
X = np.random.randn(100, 2)
correlation_matrix = np.array([[1, 0.6], [0.6, 0.6]])
L = np.linalg.cholesky(correlation_matrix)
X = X @ L.T
X = X - np.mean(X, axis=0)

# Calculate eigenvectors and eigenvalues
cov_matrix = np.cov(X.T)
eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)

# Sort eigenvalues and eigenvectors in descending order
idx = eigenvals.argsort()[::-1]
eigenvals = eigenvals[idx]
eigenvecs = eigenvecs[:, idx]

def analyze_data():
    """Analyze the generated data"""
    print("Data Analysis:")
    print(f"Data shape: {X.shape}")
    print(f"Mean: {np.mean(X, axis=0)}")
    print(f"Standard deviation: {np.std(X, axis=0)}")
    print(f"Correlation coefficient: {pearsonr(X[:, 0], X[:, 1])[0]:.3f}")
    print(f"Eigenvalues: {eigenvals}")
    print(f"Explained variance ratio: {eigenvals / np.sum(eigenvals)}")
    print()

def create_static_visualization():
    """Create comprehensive static PCA visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original data
    ax1.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.7, s=50)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Original Data')
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.axis('equal')
    ax1.set_xlim(-3.5, 3.5)
    ax1.set_ylim(-3.5, 3.5)
    
    # Principal components
    ax2.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.7, s=50)
    # Plot principal component directions
    for i, (eigenval, eigenvec) in enumerate(zip(eigenvals, eigenvecs.T)):
        color = 'red' if i == 0 else 'green'
        ax2.arrow(0, 0, eigenvec[0] * 3, eigenvec[1] * 3, 
                  head_width=0.1, head_length=0.1, fc=color, ec=color, linewidth=2)
        ax2.text(eigenvec[0] * 3.2, eigenvec[1] * 3.2, f'PC{i+1}', 
                 fontsize=12, ha='center', va='center', color=color)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Principal Components')
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    ax2.axis('equal')
    ax2.set_xlim(-3.5, 3.5)
    ax2.set_ylim(-3.5, 3.5)
    
    # Projection onto first PC
    pc1_projection = X @ eigenvecs[:, 0].reshape(-1, 1) @ eigenvecs[:, 0].reshape(1, -1)
    ax3.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.7, s=50)
    ax3.scatter(pc1_projection[:, 0], pc1_projection[:, 1], c='red', alpha=0.7, s=50)
    # Draw projection lines
    for i in range(100):
        ax3.plot([X[i, 0], pc1_projection[i, 0]], [X[i, 1], pc1_projection[i, 1]], 
                 'r', alpha=0.3, linewidth=0.5)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Projection onto First Principal Component')
    ax3.set_xlabel('X1')
    ax3.set_ylabel('X2')
    ax3.axis('equal')
    ax3.set_xlim(-3.5, 3.5)
    ax3.set_ylim(-3.5, 3.5)
    
    # Explained variance
    explained_var = eigenvals / np.sum(eigenvals)
    ax4.bar(range(1, len(explained_var) + 1), explained_var, color=['red', 'green'])
    ax4.set_title('Explained Variance Ratio')
    ax4.set_xlabel('Principal Component')
    ax4.set_ylabel('Explained Variance Ratio')
    ax4.set_xticks(range(1, len(explained_var) + 1))
    for i, v in enumerate(explained_var):
        ax4.text(i + 1, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('pca_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_rotating_animation_clean():
    """Create clean rotating PCA animation without variance tracking"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    def animate(alpha):
        ax.clear()
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Calculate projection direction
        w = np.array([np.cos(np.radians(alpha)), np.sin(np.radians(alpha))])
        z = X @ w.reshape(-1, 1) @ w.reshape(1, -1)
        
        # Draw projection lines
        for i in range(100):
            ax.plot([X[i, 0], z[i, 0]], [X[i, 1], z[i, 1]], 'r', alpha=0.3, linewidth=0.5)
        
        # Draw projection line
        ax.plot(w[0] * 3.5 * np.array([-1, 1]), w[1] * 3.5 * np.array([-1, 1]), 'k', linewidth=2)
        
        # Plot points
        ax.scatter(z[:, 0], z[:, 1], c='red', s=50, alpha=0.7)
        ax.scatter(X[:, 0], X[:, 1], c='blue', s=50, alpha=0.7)
        
        # Plot origin
        origin = Circle((0, 0), 0.1, color='white', ec='black', linewidth=2)
        ax.add_patch(origin)
        
        # Plot principal component directions
        for i, (eigenval, eigenvec) in enumerate(zip(eigenvals, eigenvecs.T)):
            color = 'red' if i == 0 else 'green'
            ax.arrow(0, 0, eigenvec[0] * 3, eigenvec[1] * 3, 
                      head_width=0.1, head_length=0.1, fc=color, ec=color, linewidth=2)
        
        ax.set_title(f'PCA Rotation Animation (Angle: {alpha}°)')
        
        return ax,
    
    anim = animation.FuncAnimation(fig, animate, frames=range(0, 181, 2), 
                                  interval=50, blit=False, repeat=True)
    
    try:
        anim.save('pca_rotating_clean.gif', writer='pillow', fps=20)
        print("Clean animation saved as 'pca_rotating_clean.gif'")
    except Exception as e:
        print(f"Could not save animation: {e}")
    
    plt.show()
    return anim

def create_pendulum_animation_clean():
    """Create clean pendulum animation without energy tracking"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Initial conditions
    alpha = -45
    omega = 0
    target_angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    
    def animate_pendulum(frame):
        nonlocal alpha, omega
        
        ax.clear()
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Calculate projection direction
        w = np.array([np.cos(np.radians(alpha)), np.sin(np.radians(alpha))])
        z = X @ w.reshape(-1, 1) @ w.reshape(1, -1)
        
        # Pendulum physics
        M = np.sum(np.sum((z @ np.array([[0, 1], [-1, 0]])) * (X - z), axis=1))
        omega = omega + M
        omega = omega * 0.93
        alpha = alpha + omega / 40
        
        # Draw projection lines
        for i in range(100):
            ax.plot([X[i, 0], z[i, 0]], [X[i, 1], z[i, 1]], 'r', alpha=0.3, linewidth=0.5)
        
        # Draw projection line
        ax.plot(w[0] * 3.5 * np.array([-1, 1]), w[1] * 3.5 * np.array([-1, 1]), 'k', linewidth=2)
        
        # Plot points
        ax.scatter(z[:, 0], z[:, 1], c='red', s=50, alpha=0.7)
        ax.scatter(X[:, 0], X[:, 1], c='blue', s=50, alpha=0.7)
        
        # Plot origin
        origin = Circle((0, 0), 0.1, color='white', ec='black', linewidth=2)
        ax.add_patch(origin)
        
        # Plot principal component directions
        for i, (eigenval, eigenvec) in enumerate(zip(eigenvals, eigenvecs.T)):
            color = 'red' if i == 0 else 'green'
            ax.arrow(0, 0, eigenvec[0] * 3, eigenvec[1] * 3, 
                      head_width=0.1, head_length=0.1, fc=color, ec=color, linewidth=2)
        
        ax.set_title(f'PCA Pendulum Animation (Angle: {alpha:.1f}°, Target: {target_angle:.1f}°)')
        
        # Stop animation if pendulum has settled
        if abs(omega) < 1 and abs(alpha - target_angle) < 1:
            plt.close()
            return ax,
        
        return ax,
    
    anim = animation.FuncAnimation(fig, animate_pendulum, frames=1000, 
                                  interval=20, blit=False, repeat=False)
    
    try:
        anim.save('pca_pendulum_clean.gif', writer='pillow', fps=50)
        print("Clean pendulum animation saved as 'pca_pendulum_clean.gif'")
    except Exception as e:
        print(f"Could not save animation: {e}")
    
    plt.show()
    return anim

if __name__ == "__main__":
    print("Creating Clean PCA animations...")
    
    # Analyze data
    analyze_data()
    
    # Create comprehensive static visualization
    print("1. Creating comprehensive static visualization...")
    create_static_visualization()
    
    # Create clean rotating animation
    print("2. Creating clean rotating animation...")
    rotating_anim = create_rotating_animation_clean()
    
    # Create clean pendulum animation
    print("3. Creating clean pendulum animation...")
    #pendulum_anim = create_pendulum_animation_clean()
    
    print("\nClean animations completed!")
    print("Files created:")
    print("- pca_comprehensive_analysis.png")
    print("- pca_rotating_clean.gif")
    print("- pca_pendulum_clean.gif")