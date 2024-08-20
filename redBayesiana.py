import pyAgrum as gum

bn = gum.BayesNet()

tl = bn.add(gum.LabelizedVariable('tiempoLibre','tiempoLibre?',2))
vp = bn.add(gum.LabelizedVariable('veoPelicula','veoPelicula',2))
vs = bn.add(gum.LabelizedVariable('veoSerie','veoSerie',2))
mn = bn.add(gum.LabelizedVariable('miroNetflix','miroNteflix',2))

bn.addArc(tl,vp)
bn.addArc(tl,vs)
bn.addArc(vp,mn)
bn.addArc(vs,mn)


bn.cpt(tl).fillWith([0.35,0.65])

bn.cpt(vp)[:]=[[0.9,0.1],[0.6,0.4]]
bn.cpt(vs)[:]=[[0.9,0.1],[0.4,0.6]]

bn.cpt(mn)[0,0,:]=[0.99,0.01]
bn.cpt(mn)[0,1,:]=[0.6,0.4]
bn.cpt(mn)[1,0,:]=[0.4,0.6]
bn.cpt(mn)[1,1,:]=[0,1]

ie=gum.LazyPropagation(bn)

print("Probabilidad total de ver netflix")

ie.makeInference()
print (ie.posterior(mn))

print ("Probabilidad de ver netflix, sabiendo que voy a ver una serie y no tengo tiempo libre")
ie.setEvidence({'veoSerie':1, 'tiempoLibre': 0})
ie.makeInference()
print (ie.posterior(mn))


print ("Probabilidad de ver netflix, sabiendo que voy a ver una serie y tengo tiempo libre")
ie.setEvidence({'veoSerie':1, 'tiempoLibre': 1})
ie.makeInference()
print (ie.posterior(mn))

print ("Probabilidad de ver netflix, sabiendo que no voy a ver una serie y no tengo tiempo libre")
ie.setEvidence({'veoSerie':0, 'tiempoLibre': 0})
ie.makeInference()
print (ie.posterior(mn))



