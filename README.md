# ConFEM2D - a finite element solver focus on reinforcement concrete 2D structures

ConFEM2D is ....

### It is based on:
	ConFEM (U Haussler-Combe, PYTHON 2)
	STAP (KJ Bathe, FORTRAN IV)
### It includes:
* Basic element types
	* Truss
	* Plate (tri, quad)
	* Shell (tri, quad)
	* Slab (generalized)
	* Spring (linear, nonlinear)
* Advanced element types
	* Multi-layers shell (tri, quad)
	* Bond element (between concrete and reinforcements)
* Multiple material types
	* Linear elastic behavior
	* Elastic behavior with limited strength (used for modeling cracked concrete)
	* Iso-damage behavior for progressive reducing stiffness after concrete's crack initializing
	* Simplified elastoplastic behavior for modeling of slab element
	* Mises's elastoplastic behavior for modeling of reinforcement
	* Nonlinear bond
* Multiple solver choices for nonlinear problem:
	* Newton-Raphson method (as default)
	* Quasi-Newton BFGS method
	* Arc length method

This table presents possible combinations between element and material types:


### It works with:
* *Gmsh4.0* as finite element mesh generator. 
* or *pygmsh* as Python interface for Gmsh

#### The documentation is available [here]() in PDF format.

### Requirements
	Python 2.7
	numpy >= 1.14
	scipy >= 1.1.0
	matplotlib >= 2.2
	scikit-umfpack >= 0.3.2
	pywin32 >= 224
	wheel >= 0.32
