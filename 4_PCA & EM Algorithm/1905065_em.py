import numpy as np
import math

def poisson_pdf(k, lambda_):
    return (lambda_**k * np.exp(-lambda_)) / math.factorial(int(k))  
        
def em_algorithm(data, max_iter=100, tolerance=1e-6):
    # Initialize parameters
    mu1 = np.mean(data) * 0.7  
    mu2 = np.mean(data) * 0.8 
    pi = 0.5  # P(C=i) where i=1 for class 1

    for iteration in range(max_iter):
        old_mu1, old_mu2, old_pi = mu1, mu2, pi
        
        # E-step: Calculate responsibilities
        resp1 = np.array([pi * poisson_pdf(x, mu1) for x in data])
        resp2 = np.array([(1-pi) * poisson_pdf(x, mu2) for x in data])
        # Normalize 
        total_resp = resp1 + resp2
        resp1 = resp1 / total_resp
        resp2 = resp2 / total_resp
        
        # M-step: Update parameters
        pi = np.mean(resp1)
        mu1 = np.sum(resp1 * data) / np.sum(resp1)
        mu2 = np.sum(resp2 * data) / np.sum(resp2)
        
        if (abs(mu1 - old_mu1) < tolerance and 
            abs(mu2 - old_mu2) < tolerance and 
            abs(pi - old_pi) < tolerance):
            break
    
    return mu1, mu2, pi, iteration+1

def main():
    data = np.loadtxt("em_data.txt")
    mu1, mu2, pi, itr = em_algorithm(data, max_iter=1000)
    if mu1 > mu2:
        mu1, mu2, pi = mu2, mu1, 1-pi    
    print(f"\nEM Results(After {itr} epoch):")
    print(f"Mean number of children in families with family planning: {mu1:.4f}")
    print(f"Mean number of children in families without family planning: {mu2:.4f}")
    print(f"Proportion of families with family planning: {pi:.4f}")
    print(f"Proportion of families without family planning: {1-pi:.4f}")

if __name__ == "__main__":
    main()
