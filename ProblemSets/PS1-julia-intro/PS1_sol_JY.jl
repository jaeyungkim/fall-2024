# adding Packages and using them
using JLD2, Random, LinearAlgebra, Plots, Statistics, CSV, DataFrames, FreqTables, Distributions

println("Current working directory: $(pwd())")
cd(dirname(@__FILE__))
println("Changed to script directory: $(pwd())")

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

# 3(a) Import nlsw88.csv and process it

println("\n" ^ 2 * "="^50)
println("QUESTION 3: Reading in Data and calculating summary statistics")
println("="^50)

println("\n3(a) Importing and processing nlsw88.csv")

# Read the raw CSV file with explicit missing value handling
nlsw88 = CSV.read("nlsw88.csv", DataFrame, 
                  missingstring=["", "NA", "NULL", ".", "missing", "n/a", " "])

println("Raw data loaded:")
println("Dimensions: $(size(nlsw88))")
println("Column names: $(names(nlsw88))")

# Check data types
println("\nColumn types:")
for col in names(nlsw88)
    println("  $col: $(eltype(nlsw88[!, col]))")
end

# Check for missing values in the raw data
println("\nMissing values per column:")
missing_summary = []
for col in names(nlsw88)
    missing_count = count(ismissing, nlsw88[!, col])
    total_count = nrow(nlsw88)
    if missing_count > 0
        pct_missing = round(100 * missing_count / total_count, digits=1)
        println("  $col: $missing_count/$total_count ($pct_missing%)")
        push!(missing_summary, (col, missing_count, pct_missing))
    else
        println("  $col: 0 missing values")
    end
end

# Clean variable names (make them more Julia-friendly)
println("\nCleaning variable names...")
original_names = names(nlsw88)

# The names look already clean, but let's make sure they're lowercase and standardized
cleaned_names = [lowercase(string(col)) for col in names(nlsw88)]

# Apply cleaned names
rename!(nlsw88, Symbol.(cleaned_names))
println("Variable names after cleaning: $(names(nlsw88))")

# Additional data validation
println("\nData validation:")
println("Sample of first 5 rows:")
display(first(nlsw88, 5))

# Check ranges for key variables to spot any obvious data issues
println("\nData ranges for key numeric variables:")
numeric_cols = ["age", "grade", "wage", "hours", "ttl_exp", "tenure"]
for col in numeric_cols
    if col in names(nlsw88)
        col_data = skipmissing(nlsw88[!, col])
        if !isempty(col_data)  # Use isempty instead of length
            min_val = minimum(col_data)
            max_val = maximum(col_data)
            mean_val = round(mean(col_data), digits=2)
            println("  $col: [$min_val, $max_val], mean = $mean_val")
        else
            println("  $col: all values are missing")
        end
    end
end

# Save the processed data
CSV.write("nlsw88_processed.csv", nlsw88)
println("\nProcessed data saved as 'nlsw88_processed.csv'")

# Final summary
println("\nFinal processed dataset summary:")
println("Dimensions: $(size(nlsw88))")
println("Variables: $(join(names(nlsw88), ", "))")
if !isempty(missing_summary)
    println("Variables with missing values: $(length(missing_summary))")
    for (col, count, pct) in missing_summary
        println("  $col: $pct%")
    end
else
    println("No missing values detected")
end

# 3(b) Calculate percentages for marital status and education
println("\n 3(b) Calculating sample percentages")

total_obs = nrow(nlsw88)
println("Total observations: $total_obs")

# Never been married percentage
# Using the 'never_married' column (1 = never married, 0 = ever married)
never_married_count = count(x -> !ismissing(x) && x == 1, nlsw88.never_married)
never_married_missing = count(ismissing, nlsw88.never_married)
never_married_valid = total_obs - never_married_missing

# Calculate percentage based on valid responses
never_married_pct = round(100 * never_married_count / never_married_valid, digits=2)
println("\nNever been married:")
println("  Count: $never_married_count out of $never_married_valid valid responses")
println("  Percentage: $never_married_pct%")
if never_married_missing > 0
    println("  Missing: $never_married_missing observations")
end

# College graduates percentage
# Using the 'collgrad' column (1 = college graduate, 0 = not college graduate)
college_grad_count = count(x -> !ismissing(x) && x == 1, nlsw88.collgrad)
collgrad_missing = count(ismissing, nlsw88.collgrad)
collgrad_valid = total_obs - collgrad_missing

# Calculate percentage based on valid responses
college_grad_pct = round(100 * college_grad_count / collgrad_valid, digits=2)
println("\nCollege graduates:")
println("  Count: $college_grad_count out of $collgrad_valid valid responses")
println("  Percentage: $college_grad_pct%")
if collgrad_missing > 0
    println("  Missing: $collgrad_missing observations")
end

# Summary
println("\n" * "="^40)
println("SUMMARY:")
println("• $never_married_pct% of the sample has never been married")
println("• $college_grad_pct% of the sample are college graduates")
println("="^40)

# 3(c) Use freqtable() to report race category percentages
using FreqTables

println("\n3(c) Race category frequencies using freqtable()")

# Create frequency table for race
race_freq = freqtable(nlsw88.race)
println("Frequency table for race:")
println(race_freq)

# Calculate percentages manually from the frequency table
total_valid = sum(race_freq)
println("\nRace category percentages:")
for (category, count) in pairs(race_freq)
    percentage = round(100 * count / total_valid, digits=2)
    println("  Race $category: $count observations ($percentage%)")
end

# Alternative: Use prop() function for direct percentages
println("\nUsing prop() for direct percentages:")
race_prop = prop(freqtable(nlsw88.race))
println(race_prop)

# Check for missing values in race variable
race_missing = count(ismissing, nlsw88.race)
if race_missing > 0
    println("\nMissing values in race: $race_missing")
    println("Percentages calculated from $total_valid valid responses")
else
    println("\nNo missing values in race variable")
end

# Summary with interpretation (assuming standard coding: 1=white, 2=black, 3=other)
println("\n" * "="^50)
println("RACE DISTRIBUTION SUMMARY:")
for (category, count) in pairs(race_freq)
    percentage = round(100 * count / total_valid, digits=2)
    race_label = if category == 1
        "White"
    elseif category == 2
        "Black"  
    elseif category == 3
        "Other"
    else
        "Category $category"
    end
    println("• $race_label: $percentage%")
end
println("="^50)

# 3(d) Use describe() function to create summary statistics matrix
println("\n3(d) Creating summary statistics using describe()")

# Create summary statistics for the entire dataframe
summarystats = describe(nlsw88)
println("Summary statistics for all variables:")
display(summarystats)

# Check specifically for missing grade observations
grade_missing = count(ismissing, nlsw88.grade)
grade_total = nrow(nlsw88)
grade_valid = grade_total - grade_missing

println("\nGrade variable missing value analysis:")
println("Total observations: $grade_total")
println("Missing grade observations: $grade_missing")
println("Valid grade observations: $grade_valid")
if grade_missing > 0
    grade_missing_pct = round(100 * grade_missing / grade_total, digits=2)
    println("Percentage missing: $grade_missing_pct%")
end

# Display detailed statistics for grade variable specifically
println("\nDetailed statistics for grade variable:")
grade_stats = describe(nlsw88.grade)
display(grade_stats)

# Additional summary information
println("\n" * "="^60)
println("SUMMARY STATISTICS OVERVIEW:")
println("Total variables analyzed: $(nrow(summarystats))")
println("Variables with missing values:")
for row in eachrow(summarystats)
    if row.nmissing > 0
        println("  $(row.variable): $(row.nmissing) missing ($(round(100*row.nmissing/grade_total, digits=1))%)")
    end
end
println("="^60)

# 3(e) Show joint distribution of industry and occupation using cross-tabulation
using FreqTables

println("\n3(e) Cross-tabulation of industry and occupation")

# Create cross-tabulation between industry and occupation
crosstab = freqtable(nlsw88.industry, nlsw88.occupation)
println("Cross-tabulation: Industry (rows) × Occupation (columns)")
display(crosstab)

# Show marginal distributions
println("\nMarginal distribution - Industry:")
industry_marginal = freqtable(nlsw88.industry)
display(industry_marginal)

println("\nMarginal distribution - Occupation:")
occupation_marginal = freqtable(nlsw88.occupation)
display(occupation_marginal)

# Calculate percentages of the joint distribution
println("\nJoint distribution as percentages:")
total_valid = sum(crosstab)
crosstab_pct = prop(crosstab) * 100  # Convert to percentages
display(round.(crosstab_pct, digits=2))

# Check for missing values in both variables
industry_missing = count(ismissing, nlsw88.industry)
occupation_missing = count(ismissing, nlsw88.occupation)
total_obs = nrow(nlsw88)

println("\nMissing value analysis:")
println("Total observations: $total_obs")
println("Missing industry values: $industry_missing")
println("Missing occupation values: $occupation_missing")
println("Valid combinations used in cross-tab: $total_valid")

# Summary statistics for the cross-tabulation
println("\n" * "="^50)
println("CROSS-TABULATION SUMMARY:")
println("Industry categories: $(size(crosstab, 1))")
println("Occupation categories: $(size(crosstab, 2))")
println("Total valid combinations: $total_valid")

# Find most common combination using a different approach
max_count = maximum(crosstab)
println("Most common combination:")

# Convert to regular matrix to use findall
crosstab_matrix = Matrix(crosstab)
max_indices = findall(x -> x == max_count, crosstab_matrix)

for idx in max_indices
    # Get the actual industry and occupation labels from the NamedArray
    industry_label = names(crosstab, 1)[idx[1]]  # Row names (industries)
    occupation_label = names(crosstab, 2)[idx[2]]  # Column names (occupations)
    println("  Industry $industry_label × Occupation $occupation_label: $max_count observations")
end
println("="^50)

# 3D Visualization of Cross-tabulation
using Plots
plotlyjs()  # Use interactive backend

println("\n📊 Creating 3D visualizations of industry-occupation cross-tabulation")

# Prepare data for 3D plotting
crosstab_matrix = Matrix(crosstab)
industry_codes = names(crosstab, 1)
occupation_codes = names(crosstab, 2)

# Create coordinate matrices for 3D plotting
n_industries = length(industry_codes)
n_occupations = length(occupation_codes)

# Method 1: 3D Surface Plot
println("Creating 3D surface plot...")
p1 = surface(
    industry_codes,
    occupation_codes,
    crosstab_matrix',
    title="Industry × Occupation Distribution (3D Surface)",
    xlabel="Industry Code",
    ylabel="Occupation Code",
    zlabel="Number of People",
    camera=(45, 30),
    color=:viridis,
    fill=true
)
display(p1)

# Method 2: 3D Wireframe
println("Creating 3D wireframe...")
p2 = wireframe(
    industry_codes,
    occupation_codes,
    crosstab_matrix',
    title="Industry × Occupation (3D Wireframe)",
    xlabel="Industry Code",
    ylabel="Occupation Code",
    zlabel="Number of People",
    camera=(60, 30),
    color=:plasma
)
display(p2)

# Method 3: 3D Scatter Plot - SIMPLE VERSION
println("Creating 3D scatter plot...")

# Create data for scatter plot
x_coords = Float64[]
y_coords = Float64[]
z_coords = Float64[]

for (i, industry) in enumerate(valid_industry_codes)
    for (j, occupation) in enumerate(valid_occupation_codes)
        count = crosstab_matrix[i, j]
        if count > 0
            push!(x_coords, Float64(industry))
            push!(y_coords, Float64(occupation))
            push!(z_coords, Float64(count))
        end
    end
end

# Create the plot if we have data
if length(x_coords) > 0
    println("Found $(length(x_coords)) data points to plot")
    
    # Try different approaches
    try
        # Approach 1: Basic scatter3d
        p3a = plot(x_coords, y_coords, z_coords, 
                  seriestype=:scatter3d,
                  title="Industry × Occupation (3D Scatter)",
                  xlabel="Industry Code",
                  ylabel="Occupation Code",
                  zlabel="Number of People",
                  markersize=5,
                  color=:blue,
                  alpha=0.7)
        display(p3a)
        
    catch e1
        println("Approach 1 failed: $e1")
        
        try
            # Approach 2: Use GR backend for 3D
            gr()
            p3b = scatter(x_coords, y_coords, z_coords,
                         title="Industry × Occupation (3D Scatter - GR)",
                         xlabel="Industry Code",
                         ylabel="Occupation Code",
                         zlabel="Number of People",
                         markersize=3,
                         color=:red)
            display(p3b)
            plotlyjs()  # Switch back to plotlyjs
            
        catch e2
            println("Approach 2 failed: $e2")
            
            # Approach 3: 2D scatter with count as color
            println("Creating 2D scatter plot instead...")
            p3c = scatter(x_coords, y_coords,
                         markersize=z_coords./5,
                         color=z_coords,
                         title="Industry × Occupation (2D - Size = Count)",
                         xlabel="Industry Code", 
                         ylabel="Occupation Code",
                         colorbar=true,
                         alpha=0.7)
            display(p3c)
        end
    end
else
    println("No valid data points found")
    # Let's debug what we have
    println("Industry codes: $valid_industry_codes")
    println("Occupation codes: $valid_occupation_codes") 
    println("Crosstab matrix size: $(size(crosstab_matrix))")
    println("Non-zero elements in crosstab: $(count(crosstab_matrix .> 0))")
end

# Method 4: Enhanced Heatmap with annotations
println("Creating enhanced heatmap...")
p4 = heatmap(
    occupation_codes,
    industry_codes,
    crosstab_matrix,
    title="Industry × Occupation Heatmap",
    xlabel="Occupation Code",
    ylabel="Industry Code",
    color=:hot,
    aspect_ratio=:auto
)

# Add text annotations showing counts
for (i, industry) in enumerate(industry_codes)
    for (j, occupation) in enumerate(occupation_codes)
        count = crosstab_matrix[i, j]
        if count > 5  # Only show counts > 5 to avoid clutter
            annotate!(p4, j, i, text(string(count), 8, :white, :center))
        end
    end
end
display(p4)

# Method 5: Bar plots for top industries (DIMENSION FIX)
println("Creating industry breakdown plots...")

# Debug the dimensions first
println("Debug info:")
println("Number of industries: $n_industries")
println("Number of occupations: $n_occupations")
println("Crosstab matrix size: $(size(crosstab_matrix))")
println("Valid industry codes length: $(length(valid_industry_codes))")
println("Valid occupation codes length: $(length(valid_occupation_codes))")

try
    max_industries_to_show = min(3, n_industries)  # Start with just 3
    
    for i in 1:max_industries_to_show
        industry = valid_industry_codes[i]
        
        # Get data for this industry (row i)
        industry_data = crosstab_matrix[i, :]
        
        # Debug this specific industry
        println("\nIndustry $industry (row $i):")
        println("  Industry data length: $(length(industry_data))")
        println("  Occupation codes length: $(length(valid_occupation_codes))")
        println("  Industry data: $industry_data")
        
        # Make sure dimensions match
        if length(industry_data) == length(valid_occupation_codes)
            try
                p_industry = bar(
                    1:length(valid_occupation_codes),  # Use indices instead of codes
                    industry_data,
                    title="Industry $industry",
                    xlabel="Occupation Index",
                    ylabel="Count",
                    color=:steelblue,
                    alpha=0.7,
                    legend=false,
                    xticks=(1:length(valid_occupation_codes), string.(valid_occupation_codes))
                )
                display(p_industry)
                println("✅ Successfully plotted industry $industry")
                
            catch e
                println("❌ Plotting failed for industry $industry: $e")
            end
        else
            println("❌ Dimension mismatch for industry $industry")
            println("   Expected: $(length(valid_occupation_codes)), Got: $(length(industry_data))")
        end
    end
    
catch e
    println("Overall error: $e")
end

# Alternative: Simple working version
println("\n" * "="^50)
println("ALTERNATIVE: Creating simple summary plots")

try
    # Plot 1: Total workers by industry
    industry_totals = [sum(crosstab_matrix[i, :]) for i in 1:n_industries]
    
    p_industry_totals = bar(
        1:length(industry_totals),
        industry_totals,
        title="Total Workers by Industry",
        xlabel="Industry",
        ylabel="Total Workers",
        color=:lightblue,
        legend=false,
        xticks=(1:length(valid_industry_codes), string.(valid_industry_codes))
    )
    display(p_industry_totals)
    
    # Plot 2: Total workers by occupation
    occupation_totals = [sum(crosstab_matrix[:, j]) for j in 1:n_occupations]
    
    p_occupation_totals = bar(
        1:length(occupation_totals),
        occupation_totals,
        title="Total Workers by Occupation",
        xlabel="Occupation",
        ylabel="Total Workers",
        color=:lightgreen,
        legend=false,
        xticks=(1:length(valid_occupation_codes), string.(valid_occupation_codes))
    )
    display(p_occupation_totals)
    
    println("✅ Summary plots created successfully!")
    
catch e
    println("❌ Summary plots failed: $e")
end

# Super simple version if all else fails
println("\n" * "="^30)
println("SUPER SIMPLE VERSION:")

try
    # Just show the numbers
    println("Top 5 industry-occupation combinations:")
    
    # Find top combinations
    flat_indices = []
    flat_values = []
    
    for i in 1:n_industries
        for j in 1:n_occupations
            push!(flat_indices, (i, j))
            push!(flat_values, crosstab_matrix[i, j])
        end
    end
    
    # Sort by values
    sorted_indices = sortperm(flat_values, rev=true)
    
    for k in 1:min(5, length(sorted_indices))
        idx = flat_indices[sorted_indices[k]]
        value = flat_values[sorted_indices[k]]
        industry = valid_industry_codes[idx[1]]
        occupation = valid_occupation_codes[idx[2]]
        println("  $k. Industry $industry × Occupation $occupation: $value workers")
    end
    
catch e
    println("❌ Even simple version failed: $e")
end

# Summary of visualizations
println("\n" * "="^60)
println("3D VISUALIZATION SUMMARY:")
println("📊 Created 5 different visualizations:")
println("  1. 3D Surface Plot - Smooth surface showing distribution")
println("  2. 3D Wireframe - Mesh view of the data structure")
println("  3. 3D Scatter Plot - Bubble size = number of people")
println("  4. Enhanced Heatmap - 2D with count annotations") 
println("  5. Industry Breakdown - Separate bars for each industry")
println("\n💡 Tips:")
println("  • Click and drag to rotate the 3D plots")
println("  • Use mouse wheel to zoom in/out")
println("  • Hover over points to see exact values")
println("="^60)

# 3(f) Tabulate mean wage over industry and occupation categories
println("\n3(f) Mean wage by industry and occupation using split-apply-combine")

# Step 1: Subset the data frame to include only relevant columns
wage_subset = select(nlsw88, [:industry, :occupation, :wage])
println("Subset created with columns: $(names(wage_subset))")
println("Dimensions: $(size(wage_subset))")

# Check for missing values in our subset
println("\nMissing values in subset:")
for col in names(wage_subset)
    missing_count = count(ismissing, wage_subset[!, col])
    if missing_count > 0
        println("  $col: $missing_count missing values")
    end
end

# Step 2: Remove rows with missing values for complete cases analysis
wage_complete = dropmissing(wage_subset)
println("\nAfter removing missing values:")
println("Dimensions: $(size(wage_complete))")

# Step 3: Split-Apply-Combine approach

# Method 1: Group by industry only
println("\n📊 Mean wage by industry:")
industry_wages = combine(groupby(wage_complete, :industry), 
                        :wage => mean => :mean_wage,
                        :wage => length => :count)
display(industry_wages)

# Method 2: Group by occupation only  
println("\n📊 Mean wage by occupation:")
occupation_wages = combine(groupby(wage_complete, :occupation),
                          :wage => mean => :mean_wage,
                          :wage => length => :count)
display(occupation_wages)

# Method 3: Group by both industry and occupation (cross-tabulation of means)
println("\n📊 Mean wage by industry AND occupation:")
industry_occupation_wages = combine(groupby(wage_complete, [:industry, :occupation]),
                                   :wage => mean => :mean_wage,
                                   :wage => std => :std_wage,
                                   :wage => length => :count)
display(industry_occupation_wages)

# Step 4: Create a pivot table for better visualization
println("\n📊 Creating pivot table of mean wages:")

# Use unstack to create a cross-tabulation format
try
    wage_pivot = unstack(industry_occupation_wages, :industry, :occupation, :mean_wage)
    println("Mean wage pivot table (Industry × Occupation):")
    display(wage_pivot)
    
    # Fill missing values with "---" for display
    wage_pivot_display = copy(wage_pivot)
    for col in names(wage_pivot_display)
        if col != :industry  # Don't modify the industry column
            wage_pivot_display[!, col] = coalesce.(wage_pivot_display[!, col], "---")
        end
    end
    
    println("\nFormatted pivot table (missing combinations shown as '---'):")
    display(wage_pivot_display)
    
catch e
    println("Pivot table creation failed: $e")
    println("Showing the grouped data instead:")
    display(industry_occupation_wages)
end

# Step 5: Summary statistics
println("\n" * "="^60)
println("WAGE ANALYSIS SUMMARY:")

# Overall wage statistics
overall_mean = mean(wage_complete.wage)
overall_std = std(wage_complete.wage)
overall_min = minimum(wage_complete.wage)
overall_max = maximum(wage_complete.wage)

println("Overall wage statistics:")
println("  Mean: \$$(round(overall_mean, digits=2))")
println("  Std Dev: \$$(round(overall_std, digits=2))")
println("  Range: \$$(round(overall_min, digits=2)) - \$$(round(overall_max, digits=2))")

# Find highest and lowest paying combinations
if nrow(industry_occupation_wages) > 0
    highest_wage_idx = argmax(industry_occupation_wages.mean_wage)
    lowest_wage_idx = argmin(industry_occupation_wages.mean_wage)
    
    highest_combo = industry_occupation_wages[highest_wage_idx, :]
    lowest_combo = industry_occupation_wages[lowest_wage_idx, :]
    
    println("\nHighest paying combination:")
    println("  Industry $(highest_combo.industry) × Occupation $(highest_combo.occupation)")
    println("  Mean wage: \$$(round(highest_combo.mean_wage, digits=2))")
    println("  Number of workers: $(highest_combo.count)")
    
    println("\nLowest paying combination:")
    println("  Industry $(lowest_combo.industry) × Occupation $(lowest_combo.occupation)")
    println("  Mean wage: \$$(round(lowest_combo.mean_wage, digits=2))")
    println("  Number of workers: $(lowest_combo.count)")
end

# Industry with highest mean wage
if nrow(industry_wages) > 0
    top_industry_idx = argmax(industry_wages.mean_wage)
    top_industry = industry_wages[top_industry_idx, :]
    println("\nHighest paying industry:")
    println("  Industry $(top_industry.industry): \$$(round(top_industry.mean_wage, digits=2))")
end

# Occupation with highest mean wage
if nrow(occupation_wages) > 0
    top_occupation_idx = argmax(occupation_wages.mean_wage)
    top_occupation = occupation_wages[top_occupation_idx, :]
    println("\nHighest paying occupation:")
    println("  Occupation $(top_occupation.occupation): \$$(round(top_occupation.mean_wage, digits=2))")
end

println("="^60)

# 3(g) Wrap all code for question 3 in a function
function q3()
    using CSV, DataFrames, FreqTables, Statistics, Plots
    
    # Force Julia to work in the script's directory
    println("Current working directory: $(pwd())")
    cd(dirname(@__FILE__))
    println("Changed to script directory: $(pwd())")
    
    println("\n" ^ 2 * "="^50)
    println("QUESTION 3: Reading in Data and calculating summary statistics")
    println("="^50)
    
    # 3(a) Import nlsw88.csv and process it
    println("\n3(a) Importing and processing nlsw88.csv")
    
    # Read the raw CSV file with explicit missing value handling
    nlsw88 = CSV.read("nlsw88.csv", DataFrame, 
                      missingstring=["", "NA", "NULL", ".", "missing", "n/a", " "])
    
    println("Raw data loaded:")
    println("Dimensions: $(size(nlsw88))")
    println("Column names: $(names(nlsw88))")
    
    # Check data types
    println("\nColumn types:")
    for col in names(nlsw88)
        println("  $col: $(eltype(nlsw88[!, col]))")
    end
    
    # Check for missing values in the raw data
    println("\nMissing values per column:")
    missing_summary = []
    for col in names(nlsw88)
        missing_count = count(ismissing, nlsw88[!, col])
        total_count = nrow(nlsw88)
        if missing_count > 0
            pct_missing = round(100 * missing_count / total_count, digits=1)
            println("  $col: $missing_count/$total_count ($pct_missing%)")
            push!(missing_summary, (col, missing_count, pct_missing))
        else
            println("  $col: 0 missing values")
        end
    end
    
    # Clean variable names
    println("\nCleaning variable names...")
    original_names = names(nlsw88)
    cleaned_names = [lowercase(string(col)) for col in names(nlsw88)]
    rename!(nlsw88, Symbol.(cleaned_names))
    println("Variable names after cleaning: $(names(nlsw88))")
    
    # Additional data validation
    println("\nData validation:")
    println("Sample of first 5 rows:")
    display(first(nlsw88, 5))
    
    # Check ranges for key variables
    println("\nData ranges for key numeric variables:")
    numeric_cols = ["age", "grade", "wage", "hours", "ttl_exp", "tenure"]
    for col in numeric_cols
        if col in names(nlsw88)
            col_data = skipmissing(nlsw88[!, col])
            if !isempty(col_data)
                min_val = minimum(col_data)
                max_val = maximum(col_data)
                mean_val = round(mean(col_data), digits=2)
                println("  $col: [$min_val, $max_val], mean = $mean_val")
            else
                println("  $col: all values are missing")
            end
        end
    end
    
    # Save the processed data
    CSV.write("nlsw88_processed.csv", nlsw88)
    println("\n✅ Processed data saved as 'nlsw88_processed.csv'")
    
    # 3(b) Calculate percentages for marital status and education
    println("\n3(b) Calculating sample percentages")
    
    total_obs = nrow(nlsw88)
    println("Total observations: $total_obs")
    
    # Never been married percentage
    never_married_count = count(x -> !ismissing(x) && x == 1, nlsw88.never_married)
    never_married_missing = count(ismissing, nlsw88.never_married)
    never_married_valid = total_obs - never_married_missing
    never_married_pct = round(100 * never_married_count / never_married_valid, digits=2)
    
    println("\nNever been married:")
    println("  Count: $never_married_count out of $never_married_valid valid responses")
    println("  Percentage: $never_married_pct%")
    
    # College graduates percentage
    college_grad_count = count(x -> !ismissing(x) && x == 1, nlsw88.collgrad)
    collgrad_missing = count(ismissing, nlsw88.collgrad)
    collgrad_valid = total_obs - collgrad_missing
    college_grad_pct = round(100 * college_grad_count / collgrad_valid, digits=2)
    
    println("\nCollege graduates:")
    println("  Count: $college_grad_count out of $collgrad_valid valid responses")
    println("  Percentage: $college_grad_pct%")
    
    # 3(c) Use freqtable() to report race category percentages
    println("\n3(c) Race category frequencies using freqtable()")
    
    race_freq = freqtable(nlsw88.race)
    println("Frequency table for race:")
    println(race_freq)
    
    total_valid = sum(race_freq)
    println("\nRace category percentages:")
    for (category, count) in pairs(race_freq)
        percentage = round(100 * count / total_valid, digits=2)
        println("  Race $category: $count observations ($percentage%)")
    end
    
    # 3(d) Use describe() function to create summary statistics matrix
    println("\n3(d) Creating summary statistics using describe()")
    
    summarystats = describe(nlsw88)
    println("Summary statistics for all variables:")
    display(summarystats)
    
    grade_missing = count(ismissing, nlsw88.grade)
    grade_total = nrow(nlsw88)
    println("\nGrade variable missing value analysis:")
    println("Missing grade observations: $grade_missing")
    
    # 3(e) Show joint distribution using cross-tabulation
    println("\n3(e) Cross-tabulation of industry and occupation")
    
    crosstab = freqtable(nlsw88.industry, nlsw88.occupation)
    println("Cross-tabulation: Industry (rows) × Occupation (columns)")
    display(crosstab)
    
    # 3D Visualization of Cross-tabulation
    plotlyjs()
    println("\n📊 Creating visualizations of industry-occupation cross-tabulation")
    
    crosstab_matrix = Matrix(crosstab)
    valid_industry_codes = collect(names(crosstab, 1))
    valid_occupation_codes = collect(names(crosstab, 2))
    
    # Simple heatmap
    try
        p_heatmap = heatmap(
            valid_occupation_codes,
            valid_industry_codes,
            crosstab_matrix,
            title="Industry × Occupation Heatmap",
            xlabel="Occupation Code",
            ylabel="Industry Code",
            color=:hot
        )
        display(p_heatmap)
    catch e
        println("Heatmap creation failed: $e")
    end
    
    # 3(f) Tabulate mean wage over industry and occupation categories
    println("\n3(f) Mean wage by industry and occupation using split-apply-combine")
    
    # Subset the data frame
    wage_subset = select(nlsw88, [:industry, :occupation, :wage])
    println("Subset created with columns: $(names(wage_subset))")
    
    # Remove missing values
    wage_complete = dropmissing(wage_subset)
    println("After removing missing values: $(size(wage_complete))")
    
    # Group by industry only
    println("\n📊 Mean wage by industry:")
    industry_wages = combine(groupby(wage_complete, :industry), 
                            :wage => mean => :mean_wage,
                            :wage => length => :count)
    display(industry_wages)
    
    # Group by occupation only  
    println("\n📊 Mean wage by occupation:")
    occupation_wages = combine(groupby(wage_complete, :occupation),
                              :wage => mean => :mean_wage,
                              :wage => length => :count)
    display(occupation_wages)
    
    # Group by both industry and occupation
    println("\n📊 Mean wage by industry AND occupation:")
    industry_occupation_wages = combine(groupby(wage_complete, [:industry, :occupation]),
                                       :wage => mean => :mean_wage,
                                       :wage => std => :std_wage,
                                       :wage => length => :count)
    display(industry_occupation_wages)
    
    # Create pivot table
    try
        wage_pivot = unstack(industry_occupation_wages, :industry, :occupation, :mean_wage)
        println("\nMean wage pivot table (Industry × Occupation):")
        display(wage_pivot)
    catch e
        println("Pivot table creation failed: $e")
    end
    
    # Summary statistics
    println("\n" * "="^50)
    println("ANALYSIS SUMMARY:")
    println("• $never_married_pct% of the sample has never been married")
    println("• $college_grad_pct% of the sample are college graduates")
    println("• $grade_missing observations have missing grade data")
    println("• Mean wage analysis completed for $(nrow(wage_complete)) complete cases")
    
    if nrow(industry_occupation_wages) > 0
        highest_wage_idx = argmax(industry_occupation_wages.mean_wage)
        highest_combo = industry_occupation_wages[highest_wage_idx, :]
        println("• Highest paying combination: Industry $(highest_combo.industry) × Occupation $(highest_combo.occupation)")
        println("  Mean wage: \$$(round(highest_combo.mean_wage, digits=2))")
    end
    println("="^50)
    
    println("\n✅ Question 3 completed!")
    
    return nothing
end

# Call all functions in order
A, B, C, D = q1()
q2(A, B, C)
q3()

# 4. Practice with functions
# (a) Load firstmatrix.jld

using JLD2

println("\n" ^ 2 * "="^50)
println("QUESTION 4: Practice with functions")
println("="^50)

println("\n4(a) Loading firstmatrix.jld2")

# Load the matrices from the JLD2 file
loaded_data = load("firstmatrix.jld2")
println("Successfully loaded firstmatrix.jld2")

# Display what was loaded
println("Contents of the file:")
for (key, value) in loaded_data
    println("  $key: $(typeof(value)), size = $(size(value))")
end

# Extract individual matrices
A_loaded = loaded_data["A"]
B_loaded = loaded_data["B"] 
C_loaded = loaded_data["C"]
D_loaded = loaded_data["D"]

# Verify the matrices loaded correctly
println("\nVerifying loaded matrices:")
println("Matrix A: $(size(A_loaded)) - first element: $(A_loaded[1,1])")
println("Matrix B: $(size(B_loaded)) - first element: $(B_loaded[1,1])")
println("Matrix C: $(size(C_loaded)) - first element: $(C_loaded[1,1])")
println("Matrix D: $(size(D_loaded)) - first element: $(D_loaded[1,1])")

println("\n ✅ Matrices A, B, C, D successfully loaded from firstmatrix.jld2")

# 4(b, c, d, e) Write a function called matrixops
println("\n4(b) Creating matrixops function")

function matrixops(A, B)
    """
    Function that performs three matrix operations on inputs A and B
    
    Inputs:
    - A: First matrix
    - B: Second matrix
    
    Outputs:
    - element_product: Element-by-element product of A and B (A .* B)
    - transpose_product: Matrix product A'B (transpose of A times B)  
    - sum_elements: Sum of all elements of A + B
    """
    
    # Check that matrices A and B have the same dimensions for element-wise operations
    if size(A) != size(B)
        error("Matrices A and B must have the same dimensions for element-wise operations")
    end
    
    # (i) Element-by-element product of A and B
    element_product = A .* B
    
    # (ii) Product A'B (transpose of A times B)
    transpose_product = A' * B
    
    # (iii) Sum of all elements of A + B
    matrix_sum = A + B
    sum_elements = sum(matrix_sum)
    
    # Return the three results
    return element_product, transpose_product, sum_elements
end

# Test the function with the loaded matrices
println("Testing matrixops function with loaded matrices A and B:")

# Call the function
result1, result2, result3 = matrixops(A_loaded, B_loaded)

# Display the results
println("\n📊 Results from matrixops function:")

println("\n(i) Element-by-element product (A .* B):")
println("Size: $(size(result1))")
println("First 3×3 elements:")
display(round.(result1[1:3, 1:3], digits=3))

println("\n(ii) Matrix product A'B:")
println("Size: $(size(result2))")
println("Note: A' is $(size(A_loaded')) and B is $(size(B_loaded)), so A'B is $(size(result2))")
println("First 3×3 elements:")
display(round.(result2[1:3, 1:3], digits=3))

println("\n(iii) Sum of all elements of A + B:")
println("Total sum: $(round(result3, digits=3))")

# Verify our results manually
println("\n🔍 Verification:")
manual_element_product = A_loaded .* B_loaded
manual_transpose_product = A_loaded' * B_loaded  
manual_sum = sum(A_loaded + B_loaded)

println("Element-wise product matches: $(result1 ≈ manual_element_product)")
println("Transpose product matches: $(result2 ≈ manual_transpose_product)")
println("Sum matches: $(result3 ≈ manual_sum)")

println("\n✅ matrixops function created and tested successfully!")

# 4(f) Evaluate matrixops using C and D from question (a) of problem 1
println("\n4(f) Evaluating matrixops with matrices C and D")

println("Matrix dimensions:")
println("C: $(size(C_loaded))")
println("D: $(size(D_loaded))")

println("\nAttempting to call matrixops(C, D):")

try
    result1, result2, result3 = matrixops(C_loaded, D_loaded)
    
    # This shouldn't execute if there's an error
    println("Function executed successfully!")
    println("Results:")
    println("  Element-wise product size: $(size(result1))")
    println("  Transpose product size: $(size(result2))")
    println("  Sum of elements: $result3")
    
catch e
    println("❌ Error occurred: $e")
    println("\nExplanation of what happened:")
    println("Matrix C has dimensions $(size(C_loaded)) (5×7)")
    println("Matrix D has dimensions $(size(D_loaded)) (10×7)")
    println("Since C and D have different dimensions, the error check in matrixops")
    println("triggered and threw an error with the message 'inputs must have the same size'")
    println("\nThis error occurs because:")
    println("• Element-wise operations like A .* B require matrices of the same size")
    println("• C is 5×7 but D is 10×7, so they cannot be multiplied element-wise")
    println("• The function correctly identified this incompatibility and stopped execution")
end

println("\n🔍 What happens:")
println("The function throws an error because C (5×7) and D (10×7) have different dimensions.")
println("This demonstrates that our error checking is working correctly!")

# 4(g) Evaluate matrixops using ttl_exp and wage from nlsw88_processed.csv
println("\n4(g) Evaluating matrixops with ttl_exp and wage from processed data")

# First, check if we have the data loaded, if not load it
if !@isdefined(nlsw88)
    println("Loading nlsw88_processed.csv...")
    nlsw88 = CSV.read("nlsw88_processed.csv", DataFrame)
    println("Data loaded successfully")
end

# Check the columns and their types
println("Checking ttl_exp and wage columns:")
println("ttl_exp type: $(eltype(nlsw88.ttl_exp))")
println("wage type: $(eltype(nlsw88.wage))")
println("ttl_exp size: $(length(nlsw88.ttl_exp))")
println("wage size: $(length(nlsw88.wage))")

# Check for missing values
ttl_exp_missing = count(ismissing, nlsw88.ttl_exp)
wage_missing = count(ismissing, nlsw88.wage)
println("Missing values - ttl_exp: $ttl_exp_missing, wage: $wage_missing")

# Convert DataFrame columns to Arrays
println("\nConverting DataFrame columns to Arrays...")
ttl_exp_array = convert(Array, nlsw88.ttl_exp)
wage_array = convert(Array, nlsw88.wage)

println("Converted arrays:")
println("ttl_exp_array type: $(eltype(ttl_exp_array))")
println("wage_array type: $(eltype(wage_array))")
println("Both arrays have length: $(length(ttl_exp_array)) and $(length(wage_array))")

# Attempt to use matrixops with these arrays
println("\nAttempting to call matrixops(ttl_exp_array, wage_array):")

try
    result1, result2, result3 = matrixops(ttl_exp_array, wage_array)
    
    println("✅ Function executed successfully!")
    println("\nResults:")
    
    println("\n(i) Element-wise product:")
    println("Size: $(size(result1))")
    println("First 5 elements: $(result1[1:5])")
    println("Type: $(eltype(result1))")
    
    println("\n(ii) Transpose product (ttl_exp' × wage):")
    println("Result: $result2")
    println("Type: $(typeof(result2))")
    println("This is a scalar because both are column vectors")
    
    println("\n(iii) Sum of all elements of ttl_exp + wage:")
    println("Total sum: $result3")
    println("Type: $(typeof(result3))")
    
    # Additional analysis
    println("\n📊 Additional analysis:")
    println("Mean of element-wise product: $(round(mean(result1), digits=3))")
    println("This represents the element-wise product of total experience and wages")
    println("The transpose product $(round(result2, digits=3)) is the dot product of the two vectors")
    println("The sum $(round(result3, digits=3)) is the total of all experience + wage values")
    
catch e
    println("❌ Error occurred: $e")
    
    # Debug information if there's an error
    println("\nDebugging information:")
    if ttl_exp_missing > 0 || wage_missing > 0
        println("Issue likely due to missing values in the data")
        println("Suggestion: Remove missing values before conversion")
        
        # Try with complete cases
        println("\nTrying with complete cases...")
        complete_data = dropmissing(nlsw88, [:ttl_exp, :wage])
        ttl_exp_clean = convert(Array, complete_data.ttl_exp)
        wage_clean = convert(Array, complete_data.wage)
        
        println("Clean data lengths: ttl_exp=$(length(ttl_exp_clean)), wage=$(length(wage_clean))")
        
        try
            result1_clean, result2_clean, result3_clean = matrixops(ttl_exp_clean, wage_clean)
            println("✅ Success with clean data!")
            println("Element-wise product size: $(size(result1_clean))")
            println("Transpose product: $(round(result2_clean, digits=3))")
            println("Sum of elements: $(round(result3_clean, digits=3))")
        catch e2
            println("❌ Still failed with clean data: $e2")
        end
    end
end

println("\n💡 Interpretation:")
println("This demonstrates using matrixops with real economic data:")
println("• ttl_exp: Total work experience (years)")
println("• wage: Hourly wage rate (dollars)")
println("• Element-wise product: Experience × wage for each person")
println("• Transpose product: Dot product of experience and wage vectors")
println("• Sum: Total of all (experience + wage) values across individuals")

# 4(h) Wrap all code for question 4 in a function
function q4()
    using JLD2, CSV, DataFrames, Statistics
    
    println("\n" ^ 2 * "="^50)
    println("QUESTION 4: Practice with functions")
    println("="^50)
    
    # 4(a) Load firstmatrix.jld2
    println("\n4(a) Loading firstmatrix.jld2")
    
    loaded_data = load("firstmatrix.jld2")
    println("Successfully loaded firstmatrix.jld2")
    
    println("Contents of the file:")
    for (key, value) in loaded_data
        println("  $key: $(typeof(value)), size = $(size(value))")
    end
    
    # Extract individual matrices
    A_loaded = loaded_data["A"]
    B_loaded = loaded_data["B"] 
    C_loaded = loaded_data["C"]
    D_loaded = loaded_data["D"]
    
    println("✅ Matrices A, B, C, D successfully loaded from firstmatrix.jld2")
    
    # 4(b) & 4(c) Define matrixops function with comment
    function matrixops(A, B)
        # This function performs three matrix operations on input matrices A and B:
        # (i) computes the element-by-element product A .* B
        # (ii) computes the matrix product A'B (transpose of A times B)  
        # (iii) computes the sum of all elements of the matrix sum A + B
        # The function returns these three results as separate outputs
        
        # Error check for input size compatibility
        if size(A) != size(B)
            error("inputs must have the same size")
        end
        
        # (i) Element-by-element product of A and B
        element_product = A .* B
        
        # (ii) Product A'B (transpose of A times B)
        transpose_product = A' * B
        
        # (iii) Sum of all elements of A + B
        matrix_sum = A + B
        sum_elements = sum(matrix_sum)
        
        # Return the three results
        return element_product, transpose_product, sum_elements
    end
    
    # 4(d) Test the function with A and B
    println("\n4(d) Testing matrixops function with loaded matrices A and B:")
    
    result1, result2, result3 = matrixops(A_loaded, B_loaded)
    
    println("\n📊 Results from matrixops function:")
    println("\n(i) Element-by-element product (A .* B):")
    println("Size: $(size(result1))")
    
    println("\n(ii) Matrix product A'B:")
    println("Size: $(size(result2))")
    
    println("\n(iii) Sum of all elements of A + B:")
    println("Total sum: $(round(result3, digits=3))")
    
    # 4(f) Evaluate matrixops using C and D
    println("\n4(f) Evaluating matrixops with matrices C and D")
    println("Matrix dimensions:")
    println("C: $(size(C_loaded))")
    println("D: $(size(D_loaded))")
    
    try
        result1_cd, result2_cd, result3_cd = matrixops(C_loaded, D_loaded)
        println("Function executed successfully!")
    catch e
        println("❌ Error occurred: $e")
        println("This error occurs because C (5×7) and D (10×7) have different dimensions.")
        println("The error checking is working correctly!")
    end
    
    # 4(g) Evaluate matrixops using ttl_exp and wage
    println("\n4(g) Evaluating matrixops with ttl_exp and wage from processed data")
    
    # Load data if not already available
    nlsw88 = CSV.read("nlsw88_processed.csv", DataFrame)
    
    # Check for missing values
    ttl_exp_missing = count(ismissing, nlsw88.ttl_exp)
    wage_missing = count(ismissing, nlsw88.wage)
    println("Missing values - ttl_exp: $ttl_exp_missing, wage: $wage_missing")
    
    # Convert DataFrame columns to Arrays
    println("Converting DataFrame columns to Arrays...")
    
    try
        # First try with original data
        ttl_exp_array = convert(Array, nlsw88.ttl_exp)
        wage_array = convert(Array, nlsw88.wage)
        
        result1_econ, result2_econ, result3_econ = matrixops(ttl_exp_array, wage_array)
        
        println("✅ Function executed successfully!")
        println("Element-wise product size: $(size(result1_econ))")
        println("Transpose product: $(round(result2_econ, digits=3))")
        println("Sum of elements: $(round(result3_econ, digits=3))")
        
    catch e
        println("❌ Error with original data: $e")
        
        if ttl_exp_missing > 0 || wage_missing > 0
            println("Trying with complete cases...")
            complete_data = dropmissing(nlsw88, [:ttl_exp, :wage])
            ttl_exp_clean = convert(Array, complete_data.ttl_exp)
            wage_clean = convert(Array, complete_data.wage)
            
            try
                result1_clean, result2_clean, result3_clean = matrixops(ttl_exp_clean, wage_clean)
                println("✅ Success with clean data!")
                println("Element-wise product size: $(size(result1_clean))")
                println("Transpose product: $(round(result2_clean, digits=3))")
                println("Sum of elements: $(round(result3_clean, digits=3))")
            catch e2
                println("❌ Still failed with clean data: $e2")
            end
        end
    end
    
    # Summary
    println("\n" * "="^50)
    println("QUESTION 4 SUMMARY:")
    println("✅ Successfully loaded matrices from firstmatrix.jld2")
    println("✅ Created matrixops function with proper error checking")
    println("✅ Tested function with compatible matrices (A, B)")
    println("✅ Demonstrated error handling with incompatible matrices (C, D)")
    println("✅ Applied function to real economic data (ttl_exp, wage)")
    println("="^50)
    
    println("\n✅ Question 4 completed!")
    
    return nothing
end

# Call all functions in order
A, B, C, D = q1()
q2(A, B, C)
q3()
q4()