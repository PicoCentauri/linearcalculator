from abc import ABC, abstractmethod

import numpy as np
from rascaline.utils import RadialIntegralFromFunction
from scipy.optimize import fsolve
from scipy.special import spherical_in, spherical_jn


def innerprod(xx, yy1, yy2):
    """
    Compute the inner product of two radially symmetric functions.

    Uses the inner product derived from the spherical integral without
    the factor of 4pi. Use simpson integration.

    Generates the integrand according to int_0^inf x^2*f1(x)*f2(x)
    """
    integrand = xx * xx * yy1 * yy2
    dx = xx[1] - xx[0]
    return (integrand[0] / 2 + integrand[-1] / 2 + np.sum(integrand[1:-1])) * dx


class RadialBasis(ABC):
    """
    Class for precomputing and storing all results related to the radial basis.

    These include:
    * A routine to evaluate the radial basis functions at the desired points
    * The transformation matrix between the orthogonalized and primitive
      radial basis (if applicable).

    Parameters
    ----------
    radial_basis : str
        The radial basis. Currently implemented are
        'GTO_primitive', 'GTO', 'monomial'.
        For monomial: Only use one radial basis r^l for each angular
        channel l leading to a total of (lmax+1)^2 features.
    max_angular : int
        Number of angular functions
    max_radial : int
        Number of radial functions
    projection_radius : float
        Environment cutoff (Å)
    orthonormalization_radius : float
        Environment cutoff (Å)
    basis_parameters: dict
        Dictionary of potential additional parameters (if needed
        for the chosen type of radial basis)
    """

    def __init__(
        self,
        radial_basis,
        max_angular,
        max_radial,
        projection_radius,
        orthonormalization_radius=None,
        basis_parameters=None,
    ):
        # Store the provided hyperparameters
        self.radial_basis = radial_basis.lower()
        self.max_angular = max_angular
        self.max_radial = max_radial
        self.projection_radius = projection_radius
        self.basis_parameters = basis_parameters

        # If the radius for the orthonormalization step is not explicitly
        # specified, use the same one as for the projection
        if orthonormalization_radius is None:
            self.orthonormalization_radius = projection_radius
        else:
            self.orthonormalization_radius = orthonormalization_radius

        # Orthonormalize
        self.compute_orthonormalization_matrix()

        # Compute nodes and weights for numerical integration
        self.define_integration_nodes_and_weights()

    def evaluate_primitive_basis_functions(self, radii):
        """
        Evaluate the basis functions prior to orthonormalization on a set
        of specified points xx.

        Parameters
        ----------
        radii : np.ndarray
            Radii on which to evaluate the (radial) basis functions

        Returns
        -------
        primitive_bases : np.ndarray
            Radial basis functions evaluated on the provided points xx.

        """
        primitive_bases = np.zeros((self.max_angular + 1, self.max_radial, len(radii)))

        # Evaluate the basis functions for all supported
        # radial bases.
        if self.radial_basis in ["gto", "gto_primitive", "gto_normalized"]:
            # Generate length scales sigma_n for R_n(x)
            sigma = np.ones(self.max_radial, dtype=float)
            for i in range(1, self.max_radial):
                sigma[i] = np.sqrt(i)
            sigma *= self.projection_radius / self.max_radial

            # Define primitive GTO-like radial basis functions
            def f_gto(n, x):
                return x**n * np.exp(-0.5 * (x / sigma[n]) ** 2)

            R_n = np.array(
                [f_gto(n, radii) for n in range(self.max_radial)]
            )  # nmax x Nradial

            # In this case, all angular channels use the same radial basis
            for l in range(self.max_angular + 1):
                primitive_bases[l] = R_n

        # Naive monomial basis consisting of functions r^n,
        # where n runs from 0,1,2,...,num_radial-1.
        # This form does not exploit the fact that we are only looking
        # at the radial dependence of a function defined in 3D space,
        # which is why "monomial_spherical" is recommended over this.
        elif self.radial_basis == "monomial_full":
            for l in range(self.max_angular + 1):
                for n in range(self.max_radial):
                    primitive_bases[l, n] = radii**n

        # Monomial basis consisting of functions R_nl(r) = r^{l+2n},
        # where n runs from 0,1,...,num_radial-1.
        # These capture precisely the radial dependence if we compute
        # the Taylor expansion of a generic funct m-lgion defined in 3D space.
        elif self.radial_basis == "monomial_spherical":
            for l in range(self.max_angular + 1):
                for n in range(self.max_radial):
                    primitive_bases[l, n] = radii ** (l + 2 * n)

        # Spherical Bessel functions used in the Laplacian eigenstate (LE)
        # basis.
        elif self.radial_basis == "spherical_bessel":
            for l in range(self.max_angular + 1):
                # Define target function and the estimated location of the
                # roots obtained from the asymptotic expansion of the
                # spherical Bessel functions for large arguments x
                def f(x):
                    return spherical_jn(l, x)

                roots_guesses = np.pi * (np.arange(1, self.max_radial + 1) + l / 2)

                # Compute roots from initial guess using Newton method
                for n, root_guess in enumerate(roots_guesses):
                    root = fsolve(f, root_guess)[0]
                    primitive_bases[l, n] = spherical_jn(
                        l, radii * root / self.self.projection_radius
                    )

        else:
            raise ValueError("The chosen radial basis is not supported!")

        return primitive_bases

    def compute_orthonormalization_matrix(self, num_nodes_ortho=50000):
        """
        Compute orthonormalization matrix for the specified radial basis

        Parameters
        ----------
        num_nodes_ortho : int, optional
            Number of nodes to be used in the numerical integration.

        Returns
        -------
        None.
        It stores the precomputed orthonormalization matrix as part of the
        class for later use, namely when calling
        "evaluate_radial_basis_functions"

        """
        # Evaluate radial basis functions
        radii = np.linspace(0, self.orthonormalization_radius, num_nodes_ortho)
        primitive_bases = self.evaluate_primitive_basis_functions(radii)

        # Gram matrix (also called overlap matrix or inner product matrix)
        innerprods = np.zeros((self.max_angular + 1, self.max_radial, self.max_radial))
        for l in range(self.max_angular + 1):
            for n1 in range(self.max_radial):
                for n2 in range(self.max_radial):
                    innerprods[l, n1, n2] = innerprod(
                        radii, primitive_bases[l, n1], primitive_bases[l, n2]
                    )

        # Get the normalization constants from the diagonal entries
        self.normalizations = np.zeros((self.max_angular + 1, self.max_radial))
        for l in range(self.max_angular + 1):
            for n in range(self.max_radial):
                self.normalizations[l, n] = 1 / np.sqrt(innerprods[l, n, n])

                # Rescale orthonormalization matrix to be defined
                # in terms of the normalized (but not yet orthonormalized)
                # basis functions
                innerprods[l, n, :] *= self.normalizations[l, n]
                innerprods[l, :, n] *= self.normalizations[l, n]

        # Compute orthonormalization matrix
        self.transformations = np.zeros(
            (self.max_angular + 1, self.max_radial, self.max_radial)
        )
        for l in range(self.max_angular + 1):
            eigvals, eigvecs = np.linalg.eigh(innerprods[l])
            self.transformations[l] = (
                eigvecs @ np.diag(np.sqrt(1.0 / eigvals)) @ eigvecs.T
            )

    def evaluate_radial_basis_functions(self, nodes):
        """
        Evaluate the orthonormalized basis functions at specified nodes.

        Parameters
        ----------
        nodes : np.ndarray of shape (N,)
            Points (radii) at which to evaluate the basis functions.

        Returns
        -------
        orthonormal_bases : np.ndarray of shape (lmax+1, nmax, N,)
            Values of the orthonormalized radial basis functions at each
            of the provided points (nodes).

        """

        # Evaluate the primitive basis functions
        primitive_bases = self.evaluate_primitive_basis_functions(nodes)

        # Convert to normalized form
        normalized_bases = primitive_bases.copy()
        for l in range(self.max_angular + 1):
            for n in range(self.max_radial):
                normalized_bases[l, n] *= self.normalizations[l, n]

        # Convert to orthonormalized form
        orthonormal_bases = np.zeros_like(primitive_bases)
        for l in range(self.max_angular + 1):
            orthonormal_bases[l] = self.transformations[l] @ normalized_bases[l]

        return orthonormal_bases

    def define_integration_nodes_and_weights(self, num_nodes=5000):
        """
        Define all nodes and weights for the numerical integration,
        which will be used to compute the radial integral.

        Instead of integrating from 0 to infinity, we truncate the
        integral up to a finite cutoff, depending on the radial basis:
        1. If the radial basis is only defined on the interval [0, rcut],
           where rcut is the cutoff radius, we only perform a radial integral
           up to rcut.
        2. If the radial basis is natively defined on [0, infinity],
           e.g. for GTOs, we use the same upper integration bound that was
           used during the orthonormalization of the radial basis.
        """
        # For now, use the composite Simpson's rule
        self.nodes = np.linspace(0, self.orthonormalization_radius, 2 * num_nodes)
        dx = self.nodes[1] - self.nodes[0]
        self.weights = np.ones_like(self.nodes)
        self.weights[::2] = 4 / 3
        self.weights[1::2] = 2 / 3
        self.weights[0] = self.weights[-1] = 1 / 3
        self.weights *= dx

    @abstractmethod
    def _radial_integral(self, n, l, k, derivative):
        """
        Custom function adapted to the "generate_splines" function
        in rascaline to compute the radial integrals

        INPUTS:
        n : int
            index of radial channel
        l : int
            index of angular channel
        k : float, np.ndarray
            k-vector
        """
        ...

    def spline_points(self, cutoff_radius, requested_accuracy=1e-8):
        def radial_basis(n, l, k):
            return self._radial_integral(n, l, k, derivative=False)

        def radial_basis_derivatives(n, l, k):
            return self._radial_integral(n, l, k, derivative=True)

        return RadialIntegralFromFunction(
            radial_basis=radial_basis,
            radial_basis_derivatives=radial_basis_derivatives,
            max_radial=self.max_radial,
            max_angular=self.max_angular,
            cutoff_radius=cutoff_radius,
            requested_accuracy=requested_accuracy,
        ).compute()


class KspaceRadialBasis(RadialBasis):
    """k/fourier space version for the radial integral."""

    def _radial_integral(self, n, l, k, derivative):
        """
        Custom function adapted to the "generate_splines" function
        in rascaline to compute the radial integrals

        I_nl(k) = int_0^infty r^2 * R_nl(r) * j_l(kr) dr

        INPUTS:
        n : int
            index of radial channel
        l : int
            index of angular channel
        k : float, np.ndarray
            k-vector
        """
        # Get integration nodes
        rr = self.nodes

        # Evaluate radial basis function for specified (n,l)
        # at these nodes.
        # WARNING: Terribly inefficient for now, since the basis
        # functions are computed for all (l,n) pairs and mostly
        # discarded at the end.
        # Ideally, we should modify the splining function so that
        # it is possible to pass all (n,l) values at once.
        R_nl = self.evaluate_radial_basis_functions(rr)[l, n]

        if type(k) in [float, int]:
            k = np.array([k])

        k = k.reshape(-1, 1)

        if derivative:
            power = 3
        else:
            power = 2

        # Compute the integral
        integrand = R_nl * rr**power * spherical_jn(l, k * rr, derivative=derivative)

        return np.sum(self.weights * integrand, axis=1)


class RspaceRadialBasis(RadialBasis):
    """real space version for the radial integral."""

    def __init__(
        self,
        radial_basis,
        max_angular,
        max_radial,
        atomic_gaussian_width,
        projection_radius,
        orthonormalization_radius=None,
        basis_parameters=None,
    ):
        self.atomic_gaussian_width = atomic_gaussian_width
        super().__init__(
            radial_basis=radial_basis,
            max_radial=max_radial,
            max_angular=max_angular,
            projection_radius=projection_radius,
            orthonormalization_radius=orthonormalization_radius,
            basis_parameters=basis_parameters,
        )

    def _radial_integral(self, n, l, rij, derivative):
        """
        Custom function adapted to the "generate_splines" function
        in rascaline to compute the radial integrals

        I_nl(rij) ∝ exp{-rij^2 / (2 σ^2)}
                    x int_0^∞ dr * r^2 * dr R_nl exp{-r^2 / (2 σ^2)} il(r rij / σ^2)

        where il is a modified spherical Bessel function.

        INPUTS:
        n : int
            index of radial channel
        l : int
            index of angular channel
        rij : float, np.ndarray
            distance vector
        """

        rr = self.nodes
        R_nl = self.evaluate_radial_basis_functions(rr)[l, n]

        if type(rij) in [float, int]:
            rij = np.array([rij])

        prefac = (
            4
            * np.pi
            / (np.pi * self.atomic_gaussian_width**2) ** (3 / 4)
            * np.exp(-(rij**2) / (2 * self.atomic_gaussian_width**2))
        )

        bessel_kernel = rr * rij.reshape(-1, 1) / self.atomic_gaussian_width**2

        integrand = (
            rr**2
            * R_nl
            * np.exp(-(rr**2) / (2 * self.atomic_gaussian_width**2))
            * spherical_in(l, bessel_kernel)
        )

        Inl = prefac * np.sum(self.weights * integrand, axis=1)

        if not derivative:
            return Inl
        else:
            dintegrand = (
                rr**3
                * R_nl
                * np.exp(-(rr**2) / (2 * self.atomic_gaussian_width**2))
                * spherical_in(l, bessel_kernel, derivative=True)
            )

            return self.atomic_gaussian_width**-2 * (
                (prefac * np.sum(self.weights * dintegrand, axis=1)) - rij * Inl
            )
