# 🎯 Julia VSCode Test Script
using LinearAlgebra, Random, Printf

println("🚀 Julia VSCode Test Suite Starting...")

# Test 1: Basic math and Unicode support
println("\n📊 Testing Unicode & Math:")
α, β = 3.14, 2.71
result = α^2 + β^2
@printf "α² + β² = %.3f\n" result

# Test 2: Array operations and broadcasting
println("\n🎲 Testing Arrays & Broadcasting:")
Random.seed!(42)
matrix = rand(3, 3)
println("Random 3×3 matrix:")
display(matrix)
scaled = matrix .* 10 .+ 1
println("Scaled matrix (.*10 .+1):")
display(round.(scaled, digits=2))

# Test 3: Function definition with multiple dispatch
println("\n⚡ Testing Multiple Dispatch:")
greet(name::String) = "Hello, " * name * "! 👋"
greet(n::Number) = "Number " * string(n) * " is not a name! 🤔"
greet(items::Vector) = "Got " * string(length(items)) * " items: " * join(items, ", ")

println(greet("Julia Programmer"))
println(greet(42))
println(greet(["apples", "bananas", "oranges"]))

# Test 4: List comprehension and filtering
println("\n🔢 Testing Comprehensions:")
squares = [x^2 for x in 1:10 if x % 2 == 0]
println("Even squares from 1-10: $squares")

# Test 5: Plotting-ready data (without actually plotting)
println("\n📈 Generating Plot Data:")
x = 0:0.1:2π
y = sin.(x) .* cos.(x/2)
println("Generated $(length(x)) data points for sin(x)*cos(x/2)")
println("Max value: $(round(maximum(y), digits=3))")

# Test 6: Struct and custom type
println("\n🏗️ Testing Custom Types:")
struct Particle
    x::Float64
    y::Float64
    velocity::Float64
end

using Plots
gr()  # Use GR backend - most stable

println("🎨 Creating simple plots...")

# Plot 1: Basic sine wave
x = 0:0.1:2π
y = sin.(x)
p1 = plot(x, y, 
          title="Sine Wave",
          xlabel="x", 
          ylabel="sin(x)",
          linewidth=2,
          color=:blue)
display(p1)

# Plot 2: Simple scatter
x_data = 1:10
y_data = [2, 5, 3, 8, 7, 6, 9, 4, 10, 1]
p2 = scatter(x_data, y_data,
            title="Simple Scatter",
            xlabel="X",
            ylabel="Y",
            markersize=8,
            color=:red)
display(p2)

# Plot 3: Bar chart
categories = ["A", "B", "C", "D", "E"]
values = [12, 19, 3, 17, 8]
p3 = bar(categories, values,
         title="Bar Chart",
         xlabel="Category",
         ylabel="Value",
         color=:green)
display(p3)

# Plot 4: Line plot with multiple series (avoiding

particles = [Particle(rand(), rand(), rand()*10) for _ in 1:5]
avg_velocity = sum(p.velocity for p in particles) / length(particles)
println("Created $(length(particles)) particles with avg velocity: $(round(avg_velocity, digits=2))")

println("\n✅ All tests completed! Julia + VSCode is working great! 🎉")