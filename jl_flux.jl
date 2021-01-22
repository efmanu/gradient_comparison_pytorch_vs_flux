using FLux
using NPZ

py_weigths = npzread("weights.npy");
py_input = npzread("input.npy");
py_outputs = npzread("output.npy");

f = py_input'
y = py_outputs'

m = Chain(
	Dense(2,5, relu),Dense(5,1)
	) #model

θ, re = Flux.destructure(m);

#initalize with python weigths
θ = py_weigths;

m = re(θ);

#reshape the first layer due to row major vs column major conflict
a = reshape( ps.order.data[1],1,:)
ps.order.data[1] .= reshape(a, (2,5))'

opt = Momentum(0.01, 0.9)

L(x, y) = Flux.Losses.mse(m(x), y)

#calculate gradient
gs = gradient(ps) do
	L(f,y)
end

@show gs.grads[ps[1]]



