# adding Packages and using them
using JLD2, Random, LinearAlgebra, Plots, Statistics, CSV, DataFrames, FreqTables, Distributions

# 1. Initializing variables and practice with basic matrix operations
# (a) Create the following four matrices of random numbers, setting the seed to ’1234’. Name the matrices and set the dimensions as noted

# Set the seed for reproducibility
Random.seed!(1234)

println("PS1: Econometrics with Julia - Matrix Operations")
println("=" ^ 50)

# Problem 1(a): Create four matrices of random numbers
println("\n1(a) Creating matrices with specified distributions:")

# i. A₁₀ₓ₇ - random numbers distributed U[-5,10]
A = rand(10, 7) * 15 .- 5  # Transform U[0,1] to U[-5,10]
println("\nMatrix A (10×7) - Uniform[-5,10]:")
println("Size: $(size(A))")
println("Sample values (first 3×3):")
display(round.(A[1:3, 1:3], digits=2))

# ii. B₁₀ₓ₇ - random numbers distributed N(-2, 15) [std dev = 15]
B = randn(10, 7) * 15 .- 2  # Transform N(0,1) to N(-2, 15)
println("\nMatrix B (10×7) - Normal(μ=-2, σ=15):")
println("Size: $(size(B))")
println("Sample values (first 3×3):")
display(round.(B[1:3, 1:3], digits=2))

# iii. C₅ₓ₇ - first 5 rows and first 5 columns of A, last 2 columns and first 5 rows of B
C = [A[1:5, 1:5] B[1:5, 6:7]]  # Horizontal concatenation
println("\nMatrix C (5×7) - Combined from A and B:")
println("Size: $(size(C))")
println("First 5 cols from A[1:5, 1:5], last 2 cols from B[1:5, 6:7]")
display(round.(C, digits=2))

# iv. D₁₀ₓ₇ - where Dᵢⱼ = Aᵢⱼ if Aᵢⱼ ≤ 0, or 0 otherwise
D = A .* (A .<= 0)  # Element-wise: keep A[i,j] if A[i,j] ≤ 0, else 0
println("\nMatrix D (10×7) - A values where A ≤ 0, zero otherwise:")
println("Size: $(size(D))")
println("Sample values (first 3×3):")
display(round.(D[1:3, 1:3], digits=2))
println("Non-zero elements: $(count(D .!= 0)) out of $(length(D))")

# (b) Number of elements in matrix A
num_elements = length(A)
println("\n Number of elements in A: $num_elements")

# (c) Number of unique elements in matrix D
num_unique = length(unique(D))
println("\n Number of unique elements in D: $num_unique")

# (d) Create matrix E using reshape() to vectorize B
E = reshape(B, :, 1)  # or reshape(B, 70, 1)
println("\n Matrix E (vectorized B using reshape):")
println("Size: $(size(E))")
display(E)

# Easier approach - direct vectorization
E = vec(B)
println("\n Matrix E (vectorized B using vec):")
println("Size: $(size(E))")
display(E)

# (e) Create 3-dimensional array F with A and B in third dimension
F = cat(A, B, dims=3)
println("\n 3-dimensional array F:")
println("Size: $(size(F))")
println("F[:,:,1] contains matrix A")
println("F[:,:,2] contains matrix B")
display(F)
# Think of it as: A stack of 2 matrices (like pages in a book), where the first page is A and the second page is B. The third dimension acts like the "page number."RetryClaude can make mistakes. Please double-check responses.

# (f) Use permutedims() to twist F from 10×7×2 to 2×10×7
F = permutedims(F, (3, 1, 2))
println("\n Permuted array F:")
println("Original size was: 10×7×2")
println("New size: $(size(F))")
display(F)

# (g) Create matrix G = B ⊗ C (Kronecker product)
G = kron(B, C)
println("\n Matrix G = B ⊗ C (Kronecker product):")
println("B size: $(size(B)), C size: $(size(C))")
println("G size: $(size(G))")
display(G)

# Try C ⊗ F
println("\n Trying C ⊗ F:")
try
    result = kron(C, F)
    println("C ⊗ F successful!")
    println("Result size: $(size(result))")
catch e
    println("Error: $e")
    println("This fails because F is 3-dimensional ($(size(F))) but Kronecker product requires 2D matrices")
end

# Explanation:
# Kronecker Product B ⊗ C:

# B is 10×7, C is 5×7
# G = B ⊗ C will be (10×5) × (7×7) = 50×49
# Each element of B is multiplied by the entire matrix C

# What happens with C ⊗ F:

# C is 2-dimensional (5×7)
# F is 3-dimensional (2×10×7)
# Kronecker product is only defined for 2D matrices
# Julia will throw an error because kron() doesn't work with 3D arrays

# The error occurs because: The Kronecker product is a matrix operation that requires both operands to be 2-dimensional matrices, but F is a 3-dimensional array.

# (h) Save matrices as .jld file
# Save all matrices to a .jld2 file
jldsave("matrixpractice.jld2"; A, B, C, D, E, F, G)
println("\n Matrices A, B, C, D, E, F, and G saved to 'matrixpractice.jld2'")

# Optional: Verify the save worked by listing contents (with explicit module qualification)
println("\n File contents:")
println(keys(JLD2.jldopen("matrixpractice.jld2")))

# (i) Save only matrices A, B, C, and D
jldsave("firstmatrix.jld2"; A, B, C, D)
println("\n Matrices A, B, C, and D saved to 'firstmatrix.jld2'")

# (j) Export C as a .csv file
# Transform C into a DataFrame
C_df = DataFrame(C, :auto)
println("\n Matrix C converted to DataFrame:")
println("Size: $(size(C_df))")
display(C_df)

# Export as CSV file
CSV.write("Cmatrix.csv", C_df)
println("\n Matrix C exported to 'Cmatrix.csv'")

# (l) Wrap all code in a function definition
function q1()    
    # Set the seed for reproducibility
    Random.seed!(1234)
    
    println("PS1: Econometrics with Julia - Matrix Operations")
    println("=" ^ 50)
    
    # Problem 1(a): Create four matrices of random numbers
    println("\n1(a) Creating matrices with specified distributions:")
    
    # i. A₁₀ₓ₇ - random numbers distributed U[-5,10]
    A = rand(10, 7) * 15 .- 5
    println("\nMatrix A (10×7) - Uniform[-5,10] created")
    
    # ii. B₁₀ₓ₇ - random numbers distributed N(-2, 15)
    B = randn(10, 7) * 15 .- 2
    println("Matrix B (10×7) - Normal(μ=-2, σ=15) created")
    
    # iii. C₅ₓ₇ - first 5 rows and first 5 columns of A, last 2 columns of B
    C = [A[1:5, 1:5] B[1:5, 6:7]]
    println("Matrix C (5×7) - Combined from A and B created")
    
    # iv. D₁₀ₓ₇ - A values where A ≤ 0, zero otherwise
    D = A .* (A .<= 0)
    println("Matrix D (10×7) - A values ≤ 0 created")
    
    # (b) Number of elements in A
    num_elements = length(A)
    println("\n(b) Number of elements in A: $num_elements")
    
    # (c) Number of unique elements in D
    num_unique = length(unique(D))
    println("(c) Number of unique elements in D: $num_unique")
    
    # (d) Create matrix E using reshape and vec
    E = reshape(B, :, 1)  # or vec(B)
    println("(d) Matrix E (vectorized B) created - Size: $(size(E))")
    
    # (e) Create 3D array F
    F = cat(A, B, dims=3)
    println("(e) 3D array F created - Size: $(size(F))")
    
    # (f) Permute dimensions of F
    F = permutedims(F, (3, 1, 2))
    println("(f) F permuted to size: $(size(F))")
    
    # (g) Kronecker product
    G = kron(B, C)
    println("(g) Matrix G = B ⊗ C created - Size: $(size(G))")
    
    # (h) Save matrices to .jld2 file
    jldsave("matrixpractice.jld2"; A, B, C, D, E, F, G)
    println("(h) All matrices saved to matrixpractice.jld2")
    
    # (i) Save first four matrices
    jldsave("firstmatrix.jld2"; A, B, C, D)
    println("(i) Matrices A, B, C, D saved to firstmatrix.jld2")
    
    # (j) Export C as CSV
    C_df = DataFrame(C, :auto)
    CSV.write("Cmatrix.csv", C_df)
    println("(j) Matrix C exported to Cmatrix.csv")
    
    println("\n All operations completed")
    
    # Return the four main matrices
    return A, B, C, D
end

# Execute the function
A, B, C, D = q1()

# 2(a) Element-by-element product using a loop
println("\n 2(a) Element-by-element product of A and B")

# Method 1: Using nested loops
AB = zeros(size(A))  # Initialize matrix with same size as A
for i in 1:size(A, 1)
    for j in 1:size(A, 2)
        AB[i, j] = A[i, j] * B[i, j]
    end
end
println("AB created using nested loops")
println("Size: $(size(AB))")
display(AB)

# Method 2: Using comprehension
AB = [A[i, j] * B[i, j] for i in 1:size(A, 1), j in 1:size(A, 2)]
println("\n AB created using comprehension")
println("Size: $(size(AB))")
display(AB)

# 2(b) Extract elements from C between -5 and 5 (inclusive)
println("\n 2(b) Elements of C between -5 and 5 (inclusive)")

# Method 1: Using a loop
Cprime = Float64[]  # Initialize empty vector
for i in 1:size(C, 1)
    for j in 1:size(C, 2)
        if -5 <= C[i, j] <= 5
            push!(Cprime, C[i, j])
        end
    end
end
println("Cprime created using loops:")
println("Length: $(length(Cprime))")
display(Cprime)

# Method 2: Without a loop (vectorized filtering)
Cprime2 = C[(-5 .<= C) .& (C .<= 5)]
println("\n Cprime2 created using vectorized operations:")
println("Length: $(length(Cprime2))")
display(Cprime2)

# Verify both methods give the same result
println("Same elements (ignoring order): $(sort(Cprime) ≈ sort(Cprime2))")

# 2(c) Create 3-dimensional array X
println("\n 2(c) Creating 3-dimensional array X")

# Set dimensions
N = 15169
K = 6
T = 5

# Initialize the 3D array
X = zeros(N, K, T)

println("Creating X array with dimensions: $N × $K × $T")

# Loop through each time period t
for t in 1:T
    println("Filling time period t = $t")
    
    # Column 1: Intercept (vector of ones) - stationary
    X[:, 1, t] = ones(N)
    
    # Column 2: Dummy variable with probability 0.75*(6-t)/5
    prob = 0.75 * (6 - t) / 5
    X[:, 2, t] = rand(N) .< prob  # Creates 1s and 0s based on probability
    
    # Column 3: Normal with mean 15+t-1 and std 5(t-1)
    # Note: when t=1, std=0, so we need to handle this case
    if t == 1
        X[:, 3, t] = fill(15.0, N)  # All values = mean when std = 0
    else
        mean_val = 15 + t - 1
        std_val = 5 * (t - 1)
        X[:, 3, t] = rand(Normal(mean_val, std_val), N)
    end
    
    # Column 4: Normal with mean π(6-t)/3 and std 1/e
    mean_val = π * (6 - t) / 3
    std_val = 1 / ℯ
    X[:, 4, t] = rand(Normal(mean_val, std_val), N)
    
    # Column 5: Discrete normal (Binomial(20, 0.6)) - stationary
    if t == 1  # Only generate once since it's stationary
        X[:, 5, :] = repeat(rand(Binomial(20, 0.6), N), 1, T)
    end
    
    # Column 6: Binomial(20, 0.5) - stationary  
    if t == 1  # Only generate once since it's stationary
        X[:, 6, :] = repeat(rand(Binomial(20, 0.5), N), 1, T)
    end
end

println("X array created successfully")
println("Dimensions: $(size(X))")

# Display summary statistics for verification
println("\n Summary for t=1:")
for k in 1:K
    col_name = ["Intercept", "Dummy", "Normal1", "Normal2", "Binomial1", "Binomial2"][k]
    println("Column $k ($col_name): mean = $(round(mean(X[:, k, 1]), digits=3))")
end

display(X[1:5, :, 1])  # Show first 5 rows for t=1

# 2(d) Create matrix β using comprehensions
println("\n2(d) Creating matrix β (K×T) with time-evolving elements")

K = 6
T = 5

# Create β matrix using comprehension
β = [
    if k == 1
        1 + 0.25 * (t - 1)           # Row 1: 1, 1.25, 1.5, 1.75, 2.0
    elseif k == 2  
        log(t)                       # Row 2: ln(t)
    elseif k == 3
        -sqrt(t)                     # Row 3: -√t  
    elseif k == 4
        exp(t) - exp(t + 1)         # Row 4: e^t - e^(t+1)
    elseif k == 5
        t                           # Row 5: t
    else  # k == 6
        t / 3                       # Row 6: t/3
    end
    for k in 1:K, t in 1:T
]

println("Matrix β created using comprehension:")
println("Dimensions: $(size(β))")
display(β)

# Show what each row represents
println("\nRow interpretations:")
println("Row 1 (1, 1.25, 1.5, ...): $(β[1, :])")
println("Row 2 (ln(t)): $(round.(β[2, :], digits=3))")
println("Row 3 (-√t): $(round.(β[3, :], digits=3))")
println("Row 4 (e^t - e^(t+1)): $(round.(β[4, :], digits=3))")
println("Row 5 (t): $(β[5, :])")
println("Row 6 (t/3): $(round.(β[6, :], digits=3))")

# 2(e) Create matrix Y using comprehension: Y_t = X_t * β_t + ε_t
println("\n 2(e) Creating matrix Y (N×T) using Y_t = X_t * β_t + ε_t")

# Create Y matrix using comprehension
Y = [
    # For each observation i and time period t:
    sum(X[i, k, t] * β[k, t] for k in 1:K) +  # X_t * β_t (matrix multiplication)
    rand(Normal(0, 0.36))                      # + ε_t ~ N(0, σ=0.36)
    for i in 1:N, t in 1:T
]

println("Matrix Y created using comprehension:")
println("Dimensions: $(size(Y))")
println("Y represents the dependent variable in our econometric model")

# Display summary statistics
println("\n Summary statistics for Y:")
for t in 1:T
    y_mean = mean(Y[:, t])
    y_std = std(Y[:, t])
    println("Time $t: mean = $(round(y_mean, digits=3)), std = $(round(y_std, digits=3))")
end

# Show first few observations
println("\n First 5 observations across all time periods:")
display(Y[1:5, :])

# 2(f) Wrap all code for question 2 in a function
function q2(A, B, C)
 
    println("\n" ^ 2 * "="^50)
    println("QUESTION 2: Practice with loops and comprehensions")
    println("="^50)
    
    # (a) Element-by-element product using loop and comprehension
    println("\n2(a) Element-by-element product of A and B")
    
    # Method 1: Using nested loops
    AB = zeros(size(A))
    for i in 1:size(A, 1)
        for j in 1:size(A, 2)
            AB[i, j] = A[i, j] * B[i, j]
        end
    end
    println("AB created using nested loops")
    display(AB)
    
    # Method 2: Without loop or comprehension (vectorized)
    AB2 = A .* B
    println("AB2 created using element-wise multiplication")
    display(AB2)
    println("Methods identical: $(AB ≈ AB2)")
    
    # (b) Extract elements from C between -5 and 5
    println("\n2(b) Elements of C between -5 and 5 (inclusive)")
    
    # Method 1: Using loops
    Cprime = Float64[]
    for i in 1:size(C, 1)
        for j in 1:size(C, 2)
            if -5 <= C[i, j] <= 5
                push!(Cprime, C[i, j])
            end
        end
    end
    println("Cprime created using loops (length: $(length(Cprime)))")
    
    # Method 2: Vectorized
    Cprime2 = C[(-5 .<= C) .& (C .<= 5)]
    println("Cprime2 created vectorized (length: $(length(Cprime2)))")
    println("Same elements (sorted): $(sort(Cprime) ≈ sort(Cprime2))")
    
    # (c) Create 3-dimensional array X
    println("\n2(c) Creating 3-dimensional array X")
    
    N = 15169
    K = 6
    T = 5
    X = zeros(N, K, T)
        
    for t in 1:T
        # Column 1: Intercept (stationary)
        X[:, 1, t] = ones(N)
        
        # Column 2: Dummy variable with time-varying probability
        prob = 0.75 * (6 - t) / 5
        X[:, 2, t] = rand(N) .< prob
        
        # Column 3: Normal with time-varying parameters
        if t == 1
            X[:, 3, t] = fill(15.0, N)
        else
            mean_val = 15 + t - 1
            std_val = 5 * (t - 1)
            X[:, 3, t] = rand(Normal(mean_val, std_val), N)
        end
        
        # Column 4: Normal with time-varying mean
        mean_val = π * (6 - t) / 3
        std_val = 1 / ℯ
        X[:, 4, t] = rand(Normal(mean_val, std_val), N)
        
        # Column 5: Binomial (stationary)
        if t == 1
            X[:, 5, :] = repeat(rand(Binomial(20, 0.6), N), 1, T)
        end
        
        # Column 6: Binomial (stationary)
        if t == 1
            X[:, 6, :] = repeat(rand(Binomial(20, 0.5), N), 1, T)
        end
    end
    
    println("X array created with dimensions: $(size(X))")
    
    # (d) Create matrix β using comprehensions
    println("\n2(d) Creating matrix β (K×T) with time-evolving elements")
    
    β = [
        if k == 1
            1 + 0.25 * (t - 1)           # 1, 1.25, 1.5, ...
        elseif k == 2
            log(t)                       # ln(t)
        elseif k == 3
            -sqrt(t)                     # -√t
        elseif k == 4
            exp(t) - exp(t + 1)         # e^t - e^(t+1)
        elseif k == 5
            t                           # t
        else # k == 6
            t / 3                       # t/3
        end
        for k in 1:K, t in 1:T
    ]
    
    println("Matrix β created with dimensions: $(size(β))")
    display(β)
    
    # (e) Create matrix Y using comprehension
    println("\n2(e) Creating matrix Y using Y_t = X_t * β_t + ε_t")
    
    Y = [
        sum(X[i, k, t] * β[k, t] for k in 1:K) + rand(Normal(0, 0.36))
        for i in 1:N, t in 1:T
    ]
    
    println("Matrix Y created with dimensions: $(size(Y))")
    println("First 3 observations across time periods:")
    display(Y[1:3, :])
    
    println("\n Question 2 completed")
    
    return nothing
end

# Call the functions in order
A, B, C, D = q1()
q2(A, B, C)