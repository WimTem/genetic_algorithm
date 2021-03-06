using Plots, Distributions, Random

#Doel: zoek u opdat datapunt = u*h_i + (1-u)*h_j

#Define hat function
function hat(start, stop)
    y1 = [0 for i in 1:1:start]
    y2 = [i for i in 1:1:6]
    y3 = [i for i in 5:-1:0]
    y4 = [0 for i in stop:1:20]
    return vcat(y1, y2, y3, y4)
end

u = rand()
test = u*hat(1,13) + (1-u)*hat(9,21)

p1 = plot([1:21 for i in 1:3], [hat(1,13)+hat(9,21) hat(1,13)+hat(5, 17)  hat(5,17)+hat(9,21)], lw=2, layout=(3,1), label=["H1+H2" "H1+H3" "H2+H3"])
savefig(p1, "Reference functions.pdf")

#Fitness function(Mean Squared Error)
function fitness(test, base)
return sum([(i[1] - i[2]) for i in zip(test, base)].^2)/21
end


#Produce offspring
function offspring(x)
    result = []
    for i = 1:5
        for j = 1:5
            if i < j
                push!(result, (x[i]+x[j])/2)
            end
        end
    end
    return result
end

#Mutate offspring
function mutate(x)
    d = Normal(0., 0.05)
    return [i[1] + i[2] for i in zip(x, rand(d, 10))]
end

function boundary(x::Real)
    (x < 0 ? 0 : x)
    (x > 1 ? 1 : x)
end

function genetic_algorithm(pop, data, h_i, h_j, tol=1e-3, n_iter=1000)
    sol = 0
    for i = 1:n_iter
        #Evaluate fitness of population
        scores = [fitness(data, i*h_i + (1-i)*h_j) for i in pop]
        u = pop[argmin(scores)]
        if abs(sol - u) <= tol
            println("Convergence, estimate for u: ",round(u, digits=5), " after :", i, " iterations.")
            p1 = scatter(1:1:21, sol*hat(1,13)+(1-u)*hat(9,21), label="Pred")
            p2 = plot!(1:1:21, test, label="True")
            savefig("pred_vs_true.pdf")
            return p2
        end
        #Cull population, keep best half then Produce then Mutate then clamp to (0,1)
        pop = clamp.([pop[j] for j in sortperm(scores)[1:5]] |> offspring  |> mutate, 0, 1)
        sol = u
    end
    return println("No convergence after ", n_iter, " iterations. \nBest estimate: ", sol)
end

pop = rand(10)

@time genetic_algorithm(pop, test, hat(1,13), hat(9,21), 1e-6, 1e6)

u