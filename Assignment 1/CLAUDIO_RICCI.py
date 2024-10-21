import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


'''
Code for Q2
'''
def plot_polynomial(coeffs, z_range, color='b'):
    z_min, z_max = z_range
    z = np.linspace(z_min, z_max)
    # Compute the polynomial
    y = np.polyval(coeffs[::-1], z)

    plt.plot(z, y, color=color, label='Polynomial')
    plt.title('Polynomial')
    plt.legend()
    plt.xlabel('z')
    plt.ylabel('p(z)')
    plt.show()


'''
    Code for Q3
'''
def create_dataset(coeffs, z_range, sample_size, sigma, seed=42):
    torch.manual_seed(seed)
    z_min, z_max = z_range
    # generates sample_size value between [0,1)
    z = torch.rand(sample_size)
    # Rescale
    z = z * (z_max - z_min) + z_min

    # Create matrix X = [1,z,z**2,z**3,z**4]
    X = torch.stack([torch.ones(sample_size), z, z**2, z**3, z**4], dim=1)
    y_hat = sum(coeff * z**i for i, coeff in enumerate(coeffs))

    # Add Gaussian noise
    y = y_hat + torch.normal(torch.zeros(sample_size), sigma*torch.ones(sample_size))

    return X, y


'''
Code for Q5
'''
def visualize_data(X, y, coeffs, z_range, title):
    z_min, z_max = z_range
    z = np.linspace(z_min, z_max)

    # Compute the true polynomial and shows it
    y_true = np.polyval(coeffs[::-1], z)
    plt.plot(z, y_true, color='b', label='True Polynomial')

    # Scatter plot with the generated data
    plt.scatter(X[:, 1], y, alpha=0.5, label='Generated Data')

    plt.title(title)
    plt.xlabel('z')
    plt.ylabel('p(z)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    '''
    Code for Q1
    '''
    assert np.version.version=="2.1.0"

    coeffs = np.array([1, -1, 5, -0.1, 1/30])  # [w_0, w_1, w_2, w_3, w_4]
    plot_polynomial(coeffs, (-4, 4))

    '''
    Code for Q4
    '''
    sigma = 0.5
    train_sample_size = 500
    train_seed = 0
    val_sample_size = 500
    val_seed = 1
    z_range = (-2,2)

    X_train, y_train = create_dataset(coeffs, z_range, train_sample_size, sigma, train_seed)
    X_val, y_val = create_dataset(coeffs, z_range, val_sample_size, sigma, val_seed)

    '''
    Code for Q5
    '''
    visualize_data(X_train, y_train, coeffs, z_range, "Training data")
    visualize_data(X_val, y_val, coeffs, z_range, "Validation data")

    '''
    Code for Q6
    '''
    model = nn.Linear(5, 1, False)
    loss_fn = nn.MSELoss()
    learning_rate = 0.03 # Start with 0.5 like in DL_03 but to high => to loss
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # reshape just for y beacuse are 1D. X is already a 2D tensor
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)

    n_steps=600

    train_loss_vals = []
    val_loss_vals = []

    weight_list = []

    for step in range(n_steps):
        model.train()
        # Set the gradient to 0
        optimizer.zero_grad()
        # Compute the output of the model
        y_hat = model(X_train)
        # Compute the loss
        loss = loss_fn(y_hat, y_train)

        # Compute the gradient
        loss.backward()
        # Update the parameters
        optimizer.step()
        with torch.no_grad():
            # Compute the output of the model
            y_hat_val = model(X_val)
            # Compute the loss
            loss_val = loss_fn(y_hat_val, y_val)

            val_loss_vals.append(loss_val.item())
            train_loss_vals.append(loss.item())

            weight_list.append(model.weight.flatten().tolist())

    print("Step:", step, "- Loss eval:", loss_val.item(), "- Loss train:", loss.item())

    # Get the final value of the parameters
    print("Final w:", model.weight, "Final b:\n", model.bias)


    '''
    Code for Q7
    '''
    plt.plot(range(step + 1), train_loss_vals)
    plt.plot(range(step + 1), val_loss_vals)
    plt.legend(["Training loss", "Validation loss"])
    plt.title("Train Loss vs Evaluation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss value")
    plt.show()


    '''
    Code for Q8
    '''
    z_min, z_max = z_range
    z = np.linspace(z_min, z_max)
    # Compute the true polynomial
    y_true = np.polyval(coeffs[::-1], z)

    est_weight = model.weight.flatten().tolist()
    est_coeffs = est_weight[::-1]

    # Compute the estimated polynomial
    # (1 * est_weight[4] + z * est_weight[3] + z**2 * est_weight[2] + z**3 * est_weight[1] + z**4 * est_weight[0] + est_bias[0])
    est_y = np.polyval(est_coeffs, z)

    plt.plot(z, y_true, color='b', label='True Polynomial')
    plt.plot(z, est_y, color='c', label='Estimated Polynomial')
    plt.title("True vs Estimated Polynomial")
    plt.xlabel('z')
    plt.ylabel('p(z)')
    plt.legend()
    plt.show()


    '''
    Code for Q9
    '''
    weight_array = np.array(weight_list)

    colors = ['blue', 'orange', 'green', 'red', 'pink']
    # Plot weights
    for i in range(5):
        plt.plot(np.arange(n_steps), weight_array[:, i], label=f'w'+str(i), color=colors[i], alpha=0.5)

    # Plot horizontal lanes for true weights
    for i in range(5):
        plt.axhline(y=coeffs[i], linestyle='--', label=f'True w'+str(i), color=colors[i])

    plt.xlabel('Steps')
    plt.ylabel('Parameter Value')
    plt.title('Evolution of Parameters During Training')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


    '''
    Code Bonus question
    '''
    def create_bonus_dataset(x_range, sample_size, sigma, seed=42):
        torch.manual_seed(seed)
        x_min, x_max = x_range
        x = torch.rand(sample_size)
        x = x * (x_max - x_min) + x_min

        y_hat = 2 * torch.log(x+1)+3

        # Compute y (Add Gaussian noise) with mean=0 and std_dev=sigma
        y = y_hat + torch.normal(torch.zeros(sample_size), sigma*torch.ones(sample_size))

        return x, y
        

    def visualize_bonus_data(x_true, y_true, x_gen, y_gen, label):
        plt.plot(x_true, y_true, label="True function", color="b")
        plt.scatter(x_gen, y_gen, label=label+' data', alpha=0.5)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('True function vs. data generated for '+label+' of f(x) = 2log(x + 1) + 3\nwith x interval = [-0.05,' + str(a) + ']')
        plt.legend()
        plt.show()


    a_values = [0.01, 10]
    tot_loss = []

    for a in a_values:
        train_loss_vals = []
        val_loss_vals = []
        
        x = np.linspace(-0.05, a)
        f = 2 * np.log(x+1)+3

        x_range = (-0.05, a)
        sigma = 0.5

        x_train, y_train = create_bonus_dataset(x_range, train_sample_size, sigma, train_seed)
        x_val, y_val = create_bonus_dataset(x_range, val_sample_size, sigma, val_seed)

        visualize_bonus_data(x, f, x_train, y_train, 'Training')
        visualize_bonus_data(x, f, x_val, y_val, 'Validation')

        model = nn.Linear(1, 1)
        loss_fn = nn.MSELoss()
        learning_rate = 0.01
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        x_train = x_train.reshape(-1, 1)
        x_val = x_val.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)

        n_steps=600

        for step in range(n_steps):
            model.train()
            optimizer.zero_grad()
            # Compute the output of the model
            y_hat = model(x_train)
            # Compute the loss
            loss = loss_fn(y_hat, y_train)

            # Compute the gradient
            loss.backward()
            # Update the parameters
            optimizer.step()
            with torch.no_grad():
                # Compute the output of the model
                y_hat_val = model(x_val)
                # Compute the loss
                loss_val = loss_fn(y_hat_val, y_val)

                val_loss_vals.append(loss_val.item())
                train_loss_vals.append(loss.item())
                tot_loss.append(loss_val.item())

        print("Step:", step, "- Loss eval:", loss_val.item(), "- Loss train:", loss.item())

        plt.plot(range(step + 1), train_loss_vals)
        plt.plot(range(step + 1), val_loss_vals)
        plt.legend(["Training loss", "Validation loss"])
        plt.title("Train Loss vs Evaluation Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss value")
        plt.show()

    first = tot_loss[:len(tot_loss)//2]
    second = tot_loss[len(tot_loss)//2:]

    plt.plot(range(step + 1), first)
    plt.plot(range(step + 1), second)
    plt.title("Loss a=0.01 vs Loss a=10")
    plt.legend(["Loss a=0.01", "Loss a=10"])
    plt.xlabel("Steps")
    plt.ylabel("Loss value")
    plt.show()
