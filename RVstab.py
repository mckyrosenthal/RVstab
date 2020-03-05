import numpy as np
import matplotlib.pyplot as plt
import rebound
import os
from matplotlib.ticker import FormatStrFormatter
import time
import scipy.optimize as op
from scipy.interpolate import interp1d
import warnings
import math
import scipy
import matplotlib as mpl
from cycler import cycler
colors = ['#4D4D4D','#5DA5DA', '#FAA43A', '#60BD68', '#F17CB0','#B2912F','#B276B2','#DECF3F','#F15854']
mpl.rcParams['axes.prop_cycle'] = cycler(color=colors)

mpl.rcParams['font.family'] = "serif"
mpl.rcParams['text.usetex'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.direction'] = "in"
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.direction'] = "in"
mpl.rcParams['figure.figsize'] = (6,4)
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 100

deg2rad = np.pi/180.

class RVPlanet:

    """

    Planet objects in the Radial velocity system

    Parameters
    ----------
    per
        The period of the planet in days
    mass
        The mass of the planet in Jupiter masses
    e
        The planet's eccentricity
    omega
        The planet's argument of peripase, in degrees
    pomega
        The planet's longitude of peripase, in degrees
    i
        The planet's' inclination in degrees. The default i=90 corresponds to the planet orbiting perpendicular to
        the plane of the sky.
    Omega
        The planet's longitude of ascending node, in degrees
    M
        The planet's initial mean anomaly in degrees.
    l
        The planet's initial mean longitude in degrees.


    """


    
    def __init__(self, per=365.25, mass=1, e=0, omega=None, pomega=None, i=90.,Omega=0, M=None,l=None):
        M_J = 9.5458e-4

        try:
            if omega==pomega==None:
                raise Exception("Error: Need to specify one of pomega or omega")
            elif omega!=None and pomega!=None:
                raise Exception("Error: Can't specify both pomega and omega")
            elif omega==None:
                self.pomega = pomega
                if i>=90.:
                    self.omega = Omega - pomega
                else:
                    self.omega = pomega - Omega
            else:
                self.omega = omega
                if i>=90.:
                    self.pomega = Omega - omega
                else:
                    self.pomega = omega + Omega
            if M==l==None:
                raise Exception("Error: Need to specify one of M or l")
            elif M!=None and l!=None:
                raise Exception("Error: Can't specify both M and l")
            elif M!=None and e==0:
                raise Exception("Error: M is not well defined for e=0")
            elif l==None:
                self.M = M
                if i>=90.:
                    self.l = Omega - M - self.omega
                else:
                    self.l = Omega + M + self.omega
            else:
                self.l = l
                if i>=90.:
                    self.M = Omega - l - self.omega
                else:
                    self.M = l - Omega - self.omega

        except Exception as inst:
            print(inst.args[0])
            self.failed = True
        else:
            self.failed = False
            self.per = per
            self.mass = mass*M_J
            self.e = e
            self.i = i
            self.Omega = Omega



class RVSystem(RVPlanet):

    """

    Main class for RV simulations

    Parameters
    ----------
    mstar
        Mass of the star in the planetary system
    epoch
        Time that planetary parameters in simulation are referenced to
    planets
        Array of RVplanets classes representing the planets in the system
    RV_files
        Files containing radial velocity data. The first three columns of these files should be: date in JD, measured
        velocity in m/s, and error in m/s
    offsets
        Constant velocity offset for each data set
    path_to_data
        Optional prefix pointing to the location of the datasets.
    RV_data
        Array of the data contained in RV_files sorted by day. Initialized by *sort_data()*
    coords
        Coordinate system of planets in the simulation. Must be one of: "astrocentric" or "jacobi"
    sol_type
        Method by which model velocities were calculated. Must be one of: "int" or "kep"
    JDs
        Julian days of RV measurements. Array where each element of the array are the elements of one of the datasets.
         Loaded in from RV_files by calling load_data().
    vels
        RV measurements themselves. Array where each element of the array are the elements of one of the datasets.
         Loaded in from RV_files by calling load_data().
    errs
        Errors on RV measurements. Array where each element of the array are the elements of one of the datasets.
         Loaded in from RV_files by calling load_data().
    vels_model
        Theoretical radial velocities, calculated by calling log_like. These velocities are formatted so that they correspond
        to the *vels* variable
    vels_plot
        Theoretical radial velocities as one flattened array, used for plotting purposes.

    """


    
    def __init__(self,mstar=1.0):
        
        self.mstar = mstar #Mass of central star
        self.epoch = None
        self.planets = np.array([]) #Array containing RVPlanets class
        self.RV_files = [] #Array of RV velocities, assumed to be of the form: JD, RV, error
        self.path_to_data = "" #Optional prefix that points to location of datasets
        self.RV_data=[] #Array of all data sets, organized by days. Returned by function below
        self.coords = 'astrocentric'
        self.offsets = np.zeros(len(self.RV_files))
        self.sol_type = ""
        self.JDs = []
        self.vels = []
        self.errs = []
        self.vels_model = []
        self.vels_plot = []
        self.ll = 0

    def load_data(self):

        """
        Load in data contained in self.RV_files
        """


        self.JDs = []
        self.vels = []
        self.errs = []

        for i,fname in enumerate(self.RV_files):
            assert len(self.RV_files) != 0, "No data files specified"
            assert os.path.exists(self.path_to_data + fname), "Path to data files not found"
            tmp_arr = np.loadtxt(self.path_to_data + fname)
            self.JDs.append(tmp_arr[:,0])
            self.vels.append(tmp_arr[:,1])
            self.errs.append(tmp_arr[:,2])

        self.offsets = np.zeros(len(self.RV_files)) #Array of constant velocity offsets for each data set

    
    def add_planet(self,per=365.25, mass=1, M=None, l=None, e=0, omega=None, pomega=None, i=90.,Omega=0):
        """
        Add planet to RV simulation. Angles are in degrees, planet mass is in Jupiter masses

        Parameters
        ----------
        per
            The period of the planet in days
        mass
            The mass of the planet in Jupiter masses
        e
            The planet's eccentricity
        omega
            The planet's argument of peripase, in degrees. Note you cannot pass both a longitude and argument of periapse.
        pomega
            The planet's longitude of peripase, in degrees. Note you cannot pass both a longitude and argument of periapse.
        i
            The planet's' inclination in degrees. The default i=90 corresponds to the planet orbiting perpendicular to
            the plane of the sky.
        Omega
            The planet's longitude of ascending node, in degrees
        M
            The planet's initial mean anomaly in degrees. Note you cannot pass both a mean anomaly and mean longitude.
        l
            The planet's initial mean longitude in degrees. Note you cannot pass both a mean anomaly and mean longitude.
        """

        p = RVPlanet(per=per,mass=mass,e=e,omega=omega,pomega=pomega,i=i,Omega=Omega,M=M,l=l)
        if p.failed:
            print("Error: Couldn't add planet")
        else:
            self.planets = np.append(self.planets,p)

    def rem_planet(self,i=0):
        del self.planets[i]
        self.ll = 0
        self.vels_model = []
        self.offsets = np.zeros(len(self.RV_files))

    def clear_planets(self):
        self.planets = []
        self.ll = 0
        self.vels_model = []
        self.offsets = np.zeros(len(self.RV_files))


    def ret_sim(self,epoch=None):
        """
        Initialize and return the rebound simulation associated with the RVSystem class.
        """

        if epoch is None:
            epoch = self.epoch
        assert epoch != None, "No epoch specified"
        assert self.coords in ('astrocentric', 'jacobi'), "Unrecognized coordinate system"
        sim = rebound.Simulation()
        sim.units = ('day', 'AU', 'Msun')
        sim.t = epoch #Epoch is the starting time of simulation
        sim.add(m=self.mstar,hash='star')

        if self.coords == 'jacobi': #Add planets in ascending period order with Jacobi elements
            per_arr = np.array([planet.per for planet in self.planets])
            for planet in self.planets[np.argsort(per_arr)]:
                sim.add(m=planet.mass,P=planet.per,l=planet.l*deg2rad,e=planet.e,omega=planet.omega*deg2rad,
                    inc=planet.i*deg2rad,Omega=planet.Omega*deg2rad)
        else: #Add planets with coordinates referenced to central star
            for planet in self.planets:
                sim.add(primary=sim.particles[0],m=planet.mass,P=planet.per,l=planet.l*deg2rad,e=planet.e,omega=planet.omega*deg2rad,
                    inc=planet.i*deg2rad,Omega=planet.Omega*deg2rad)

        sim.move_to_com()
        return sim

    def log_like(self,jitter=0,sol_type = 'int',max_dist = 1e100, min_dist=0):

        """
        Calculate the log likelihood for the radial velocity signal of the planets in the simulation compared to the data.

        Parameters
        ----------
        jitter
            Constant error term added in quadrature to the given errors.
        sol_type
            Method by which the model velocities are calculated. "int" uses the REBOUND N-body integrator to calculate the velocities, "kep" advances the planets along Keplerian orbits.
        max_dist
            Maximum distance a planet is allowed 

        Returns
        -------
        log_like: float
            The log likelihood of the model.

        """

        self.vels_model = []

        assert len(self.JDs)!=0 or len(self.RV_files)!=0, "No data stored and no files to load from"
        assert len(self.planets) != 0, "No planets in simulation"
        assert sol_type in ('int','kep'), "Unknown model for theoretical RVs"
        self.sol_type = sol_type
        if len(self.JDs) == 0:
            warnings.warn("Data not loaded, loading in data now")
            self.load_data()

        sim = self.ret_sim()
        sim.exit_max_distance = max_dist
        sim.exit_min_distance = min_dist

        ps = sim.particles
        planets = sim.particles[1:]
        AU_day_to_m_s = 1.731456e6

        l_like = 0

        if type(self.JDs[0]) is np.float64:
            rng = 1
            flat = True
            # jitter = [jitter]
        else:
            rng = len(self.JDs)
            flat = False
            self.offsets = np.zeros(len(self.JDs))
            
        if jitter:
            if isinstance(jitter,(np.float64,float,int)):
                jitter = np.array([jitter])
            assert len(jitter) == rng, "Mismatch between number of jitters and number of datasets"
        else:
            jitter = np.zeros(rng)

        for i in range(rng):
            if flat:
                JDs = self.JDs
                errs = self.errs
                vels = self.vels
                self.offsets = np.array([0],dtype=float)
            else:
                JDs = self.JDs[i]
                vels = self.vels[i]
                errs = self.errs[i]

            vels_model = np.zeros(len(JDs))

            if sol_type == 'int':
                for j,t in enumerate(JDs):
                    try:
                        sim.integrate(t)
                    except (rebound.Escape,rebond.Encounter) as error:
                        print(error)
                        return -np.inf
                    else:
                        vels_model[j] = -ps['star'].vz * AU_day_to_m_s
            else:
                # for i,planet in enumerate(planets):
                # t_p_arr = [self.epoch-planet.M*deg2rad/2./np.pi*planet.per + planet.per for planet in self.planets]

                for j,t in enumerate(JDs):
                    rv = 0
                    for k,planet in enumerate(self.planets):
                        if planet.i >= 90.:
                            M = planet.l - planet.omega
                        else:
                            M = planet.M
                        t_p = self.epoch-M*deg2rad/2./np.pi*planet.per + planet.per
                        f = kep_sol(t=t,t_p=t_p,e=planet.e,per=planet.per)
                        rv += RV_amp(m_star = self.mstar, m_p = planet.mass, per = planet.per, f = f, e = planet.e,
                                 omega=planet.omega)
                    vels_model[j] = rv

            self.vels_model.append(vels_model)

            err_tot = np.sqrt(errs**2.+jitter[i]**2.)

            #Optimize the offsets for the given model and measured velocities. The equations below do the optimization analytically.
            S = np.sum(err_tot**(-2.))
            off = 1./S*np.sum((vels-vels_model)/err_tot**2.)
            self.offsets[i] = off

            l_like += -np.sum(0.5*(vels-off-vels_model)**2./(err_tot**2.) + np.log(np.sqrt(2*np.pi*err_tot**2.)))

        self.ll = l_like

        return l_like

    def calc_chi2(self,jitter=0,reduced=False,non_planar = False,rms=False):

        """
        Return chi^2 statistics for the Radial velocity model. Requires calling log_like() beforehand to initialize the model velocities.

        Parameters
        ----------
        jitter
            Constant error term added in quadrature to the given errors.
        reduced: bool
            If true, returns the reduced chi^2 value, i.e. chi^2/(data_points - degrees of freedom).
        non-planar: bool
            If true, the reduced chi^2 is calculated assuming the planetary system was allowed to have mutual inclination, which adds two parameters per planet.
        rms: bool
            If true, returns the root-mean-square deviation of the data from the model.

        Returns
        -------
        chi_2: float
            A measurement of the model's goodness of fit -- either the full chi^2, the reduced chi^2, or the RMS deviation, depending on the options passed to calc_chi2

        """


        assert len(self.vels_model)!=0, "No theoretical velocities calculated, run self.log_like first"

        if type(self.JDs[0]) is np.float64:
            rng = 1
            flat = True
        else:
            rng = len(self.JDs)
            flat = False
        if jitter:
            if isinstance(jitter,(np.float64,float,int)):
                jitter = np.array([jitter])
            assert len(jitter) == rng, "Mismatch between number of jitters and number of datasets"
        else:
            jitter = np.zeros(rng)

        if rms:
            rms = 0
            points = 0
            for i in range(rng):
                if flat:
                    JDs = self.JDs
                    vels = self.vels
                    vels_model = self.vels_model
                else:
                    JDs = self.JDs[i]
                    vels = self.vels[i]
                    errs = self.errs[i]
                    vels_model = self.vels_model[i]
                rms += np.sum((vels-self.offsets[i]-vels_model)**2.)
                points += len(vels)
            return np.sqrt(rms/points)

        chi_2 = 0
        points = 0
        for i in range(rng):
            if flat:
                JDs = self.JDs
                errs = self.errs
                vels = self.vels
                vels_model = self.vels_model
            else:
                JDs = self.JDs[i]
                vels = self.vels[i]
                errs = self.errs[i]
                vels_model = self.vels_model[i]
            chi_2 += np.sum(((vels-self.offsets[i]-vels_model)**2.)/(errs**2+jitter[i]**2.))
            points += len(vels)
        if reduced:
            if jitter:
                jit_num = len(jitter)
            else:
                jit_num = 0
            if non_planar:
                dof = 7.*len(self.planets) + len(self.offsets) + jit_num
            else:
                dof = 5.*len(self.planets) + len(self.offsets) + jit_num

            return chi_2/(points-dof)
        else:
            return chi_2


    def orbit_stab(self,periods=1e4,pnts_per_period=100,outputs_per_period=1,integrator='whfast',d_min = 0,safe=True,\
                   check_dist=False,verbose=False,timing=False,plot=False,energy_err=False,save_fig=False,return_arr=False):

        """
        Check whether the planetary system in the model is stable. The default is that the system is stable if both planet's semi-major axes remain between 50% and 150% of their initial values.

        Parameters
        ----------
        periods
            Timescale on which to check for stablility, in units of the longest orbital period in the system.
        pnts_per_period
            Timestep of the simulation, in units of fraction of the smallest orbital period in the system
        outputs_per_period
            Number of times per smallest period in the system to check whether the stability conditions have been violated.
        integrator
            Integrator used for the N-body simulation. Must correspond to one of the integrators built in to REBOUND.
        d_min
            Smallest allowed distance, in AU, between particles in the simulation. System will be ruled non-stable if two particles pass within d_min
        check_dist: bool
            Check planet-star distance instead of planet's semi-major axis. System is ruled non-stable if the distance between the planet and star exceeds 150% of the planets initial apocenter distance or is less than 50% of the initial pericenter distance.
        verbose: bool
            Print out information during integration, e.g. percentage of progress.
        timing: bool
            Print out how long the integration took.
        plot: bool
            Plot distance of each planet as a function of time after simulation.
        energy_err: bool
            Print out the energy error in the simulation, i.e. the percent change in initial energy vs. final energy.
        save_fig: bool
            Save plot made after simulation
        return_arr: bool
            Return times, dist_arr variables, used to plot the planets distance, in addition to the stable boolean.

        Returns
        -------
        stable: bool
            Whether or not the planetary system passed the stability criteria.

        """


        sim = self.ret_sim(epoch=0)
        sim.integrator = integrator
        if integrator == 'whfast':
            exact = 0
            if not(safe):
                sim.ri_whfast.safe_mode = 0
        else:
            exact = 1

        sim.exit_min_distance = d_min

        per_arr = [planet.per for planet in self.planets]
        max_per = np.amax(per_arr)
        min_per = np.amin(per_arr)

        sim.dt = min_per/pnts_per_period
        t_max = max_per*periods
        Noutputs = int(t_max/min_per*outputs_per_period)
        times = np.linspace(0,t_max, Noutputs)

        ps = sim.particles[1:]

        if plot:
            dist_arr = np.zeros((len(ps),Noutputs))
        if verbose:
            print (Noutputs)
        if timing:
            start_time = time.time()
        if energy_err:
            E0 = sim.calculate_energy()

        a0 = [planet.a for planet in ps]
        e0 = [planet.e for planet in ps]

        stable = 1

        try:
            for i,t in enumerate(times): #Perform integration
                sim.integrate(t,exact_finish_time = exact)
                for k,planet in enumerate(ps):
                    if check_dist:
                        apo = a0[k]*(1.+e0[k])
                        peri = a0[k]*(1.-e0[k])
                        if planet.d > 1.5*apo or planet.d < 0.5*peri or planet.d < 0.1:
                            stable = 0
                            if verbose:
                                print (planet.d,apo,peri)
                    elif planet.a > 1.5*a0[k] or planet.a < 0.5*a0[k] or planet.a < 0.1:
                        stable = 0
                    if plot:
                        if check_dist:
                            dist_arr[k,i] = planet.d
                        else:
                            dist_arr[k,i] = planet.a
                if verbose and (i % int(Noutputs/10) == 0):
                    print ("%i%%" %math.ceil(100*i/Noutputs))

                if stable == 0:
                    break
        except rebound.Encounter as error:
            if verbose:
                print(error)
            stable = 0

        if timing:
            print ("Integration took %.2f seconds" %(time.time() - start_time))

        if energy_err:
            Ef = sim.calculate_energy()
            print( "Energy Error is %.3f%% " %(np.abs((Ef-E0)/E0*100)))


        if plot:
            fig = plt.figure(1,figsize=(11,6))
            for i in range(len(ps)):
                inds = np.nonzero(dist_arr[i])
                plt.semilogx(times[inds]/365.25,dist_arr[i][inds])

            plt.xlabel("Time [Years]",fontsize=18)
            plt.ylabel("a [AU]",fontsize=18)

            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.tick_params(axis='both', which='minor', labelsize=8)

            if save_fig:
                fig.savefig('orbit_stab.pdf',format='pdf')


        if return_arr:
            return stable, times, dist_arr
        else:
            return stable

    def stab_logprob(self,jitter=0,sol_type='int',**kwargs):
        """
        Wrapper function for log_like and orbit_stab.

        Parameters
        ----------
        jitter
            Constant error term added in quadrature to the given errors.
        sol_type
            Method by which the model velocities are calculated. "int" uses the REBOUND N-body integrator to calculate the velocities, "kep" advances the planets along Keplerian orbits.
        **kwargs
            Keyword arguments to be passed to orbit_stab

        Returns
        -------
        log_like: float
            If the system is stable (i.e. if orbit_stab == True), then the log likelihood is returned. Otherwise, the function returns -np.inf

        """

        l_like = self.log_like(jitter=jitter,sol_type=sol_type)
        stable = self.orbit_stab(**kwargs)
        if stable:
            return l_like
        else:
            return -np.inf


    def calc_vels_plot(self,JD_min,JD_max,pnts_per_period=100,sol_type='int'):

        assert sol_type in ('int','kep'), "Unknown model for theoretical RVs"

        sim = self.ret_sim()
        ps = sim.particles

        per_arr = [planet.per for planet in self.planets]
        min_per = np.amin(per_arr)

        Noutputs = int((JD_max-JD_min)/min_per*pnts_per_period)

        times = np.linspace(JD_min, JD_max, Noutputs)
        AU_day_to_m_s = 1.731456e6 #Conversion factor from Rebound units to m/s

        rad_vels = np.zeros(Noutputs)

        if sol_type == 'int':

            for i,t in enumerate(times): #Perform integration
                sim.integrate(t)
                rad_vels[i] = -ps['star'].vz * AU_day_to_m_s

        else:
            # t_p_arr = [self.epoch-planet.M*deg2rad/2./np.pi*planet.per + planet.per for planet in self.planets]

            for i,t in enumerate(times):
                rv = 0
                for j,planet in enumerate(self.planets):
                    if planet.i >= 90.:
                        M = planet.l - planet.omega
                    else:
                        M = planet.M
                    t_p = self.epoch - M*deg2rad/2./np.pi*planet.per + planet.per
                    f = kep_sol(t=t,t_p=t_p,e=planet.e,per=planet.per)
                    rv += RV_amp(m_star = self.mstar, m_p = planet.mass, per = planet.per, f = f, e = planet.e,
                                 omega=planet.omega)
                rad_vels[i] = rv

        self.vels_plot = [times,rad_vels]


    def plot_RV(self,save_plot=False,filename="tmp.pdf",plot_theo=True,plot_data=True,\
                res=False,jitter=0,ret_data=0,mark_times=[],periods=5,pnts_per_period=100,sol_type= 'int'):

        """
        Make a plot of the model RV from the planets in the simulation, the measured RV, or both.

        Parameters
        ----------
        save_plot: bool
            Save the plot after its made.
        filename
            Filename of saved plot
        plot_theo: bool
            If true, plots the model radial velocity
        plot_data: bool
            If true, plots the RV data
        res
            If true, plots the residuals between the model and data. Requires plot_theo and plot_data to both be true.
        jitter
            Add a constant error term added in quadrature to the given errors, when plotting the data.
        ret_data: bool
            Return the times,rad_vels arrays used to plot the theoretical RV curve
        mark_times
            Draw vertical lines at the indicated times
        periods
            Number of periods to plot if plot_data=False. Otherwise the theoretical data will be plotted from the first data point to the last data point.
        pnts_per_period
            Timestep for plotting theoretical RV curve, in units of the smallest orbital period in the system.
        sol_type
            Method by which the model velocities are calculated. "int" uses the REBOUND N-body integrator to calculate the velocities, "kep" advances the planets along Keplerian orbits.
        """


        if plot_theo:
            assert len(self.planets)!=0, "Can't plot theoretical curve without planets!"
        if plot_data:
            assert len(self.RV_files)!=0 or len(self.JDs)!=0, "No data stored and no files to load"
            if len(self.JDs)==0:
                self.load_data()
            if type(self.JDs[0]) is np.float64:
                rng = 1
                flat = True
            else:
                rng = len(self.JDs)
                flat = False
            if self.sol_type != sol_type and plot_theo:
                warnings.warn("Reoptimizing offsets")
                self.log_like(jitter=jitter,sol_type=sol_type)
            if jitter:
                if isinstance(jitter,(np.float64,float,int)):
                    jitter = np.array([jitter])
                assert len(jitter) == rng, "Mismatch between number of jitters and number of datasets"
            else:
                jitter = np.zeros(rng)
            if np.count_nonzero(self.offsets)==0:
                if flat:
                    self.offsets = np.array([0],dtype=float)
                else:
                    self.offsets = np.zeros(len(self.JDs))
                warnings.warn("Offsets have not been optimized, call self.log_like to optimize")
        if not(plot_data and plot_theo):
            assert res==0, "Can't plot residuals without data and planets!"

        fig = plt.figure(1,figsize=(13,7))#Plot RV
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3,rowspan=2)

        if plot_theo:

            if plot_data:
                if flat:
                    JD_max = np.amax(self.JDs)
                    JD_min = np.amin(self.JDs)
                else:
                    JD_max = max(np.amax(self.JDs[i]) for i in range(len(self.JDs)))
                    JD_min = min(np.amin(self.JDs[i]) for i in range(len(self.JDs)))
            else:
                assert self.epoch != None, "No epoch specified"
                JD_min = self.epoch
                max_per = np.amax([planet.per for planet in self.planets])
                JD_max = self.epoch+periods*max_per

            self.calc_vels_plot(JD_min,JD_max,pnts_per_period=pnts_per_period,sol_type=sol_type)

            times = self.vels_plot[0]
            rad_vels = self.vels_plot[1]
            ax1.plot(times,rad_vels,lw=2,color='#4D4D4D')

        if plot_data:
            colors = ['#5DA5DA', '#FAA43A', '#60BD68', '#F17CB0','#B2912F','#B276B2','#DECF3F','#F15854']
            ms = 8
            for i in range(rng):
                if flat:
                    JDs = self.JDs
                    errs = self.errs
                    vels = self.vels
                else:
                    JDs = self.JDs[i]
                    vels = self.vels[i]
                    errs = self.errs[i]
                yerr = np.sqrt(errs**2. + jitter[i]**2.)
                ax1.errorbar(JDs,vels-self.offsets[i],yerr = yerr,fmt='o',mec='w',ms=ms,color=colors[i])

        if not(res):
            plt.xlabel("Time [JD]",fontsize=20)
        plt.ylabel("RV [m/s]",fontsize=20)
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        if len(mark_times)!=0:
            for t in mark_times:
                ax1.axvline(x=t,linestyle='dashed',color=colors[-1],linewidth=2)

        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.tick_params(axis='both', which='minor', labelsize=8)

        if res:
            ax1.set_xticklabels([])
            ax2 = plt.subplot2grid((3, 3), (2, 0),colspan=3)
            f = scipy.interpolate.interp1d(times,rad_vels,kind='cubic')
            for i in range(rng):
                if flat:
                    JDs = self.JDs
                    errs = self.errs
                    vels = self.vels
                else:
                    JDs = self.JDs[i]
                    vels = self.vels[i]
                    errs = self.errs[i]
                delta_RV = vels - self.offsets[i] - f(JDs)
                yerr = np.sqrt(errs**2. + jitter[i]**2.)
                ax2.errorbar(JDs,delta_RV,yerr = yerr,fmt='o',color=colors[i],mec='w',ms=ms)
                ax2.axhline(y=0,linestyle='dashed',color='#4D4D4D')
                plt.xlabel("Time [JD]",fontsize=20)
                plt.ylabel(r"$\Delta$RV [m/s]",fontsize=20)
                ax2.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

            plt.tick_params(axis='both', which='major', labelsize=20)
            plt.tick_params(axis='both', which='minor', labelsize=8)

        plt.show()
        
        if save_plot:
            fig.savefig(filename,format='pdf')

        if ret_data:
            return times,rad_vels

        
    def plot_phi(self,p,q,pert_ind,test_ind,periods=1e2,pnts_per_period=100.,\
                outputs_per_period=20.,log_t = False, integrator='whfast',plot=1,return_angs=0,dots=1,\
                ):
        """
        Plot the resonant angle for two planets in the simulation.

        Parameters
        ----------

        p,q
            The order of the mean-motion resonance, i.e. the resonance is taken to be p:q
        pert_ind
            The index of the self.planets array corresponding to the perturber, i.e. the massive planet in the resonance
        test_ind
            The index of the self.planets array corresponding to the test particle in the resonance
        periods
            Number of orbital periods of the outer planet to plot the resonant angle for.
        pnts_per_period
            Timestep for the integration, in units of number of points per smallest orbital period in the system.
        outputs_per_period
            Timestep for plot, in units of number of points per largest orbital period.
        log_t: bool
            If true, plots log time on the x-axis
        integrator
            Integrator used for the N-body simulation. Must correspond to one of the integrators built in to REBOUND.
        plot: bool
            Whether or not to plot the result after the integration
        return_angs: bool
            If true, returns the array used used to calculate phi, i.e. mean longitude of the outer and inner particles, and the longitude of pericenter of the test particle.
        dots: bool
            If true, make dots at each value of phi instead of a continuous line.
        """

        sim = self.ret_sim(epoch=0)
        if integrator == 'whfast':
            exact = 0

        ps = sim.particles[1:]
        test = ps[test_ind]

        per_arr = [planet.P for planet in ps]
        if per_arr[pert_ind] > per_arr[test_ind]:
            outer = ps[pert_ind]
            inner = ps[test_ind]
        else:
            outer = ps[test_ind]
            inner = ps[pert_ind]

        t_max = outer.P*periods
        Noutputs = int(t_max/inner.P*outputs_per_period)
        times = np.linspace(0,t_max, Noutputs)

        sim.dt = inner.P/pnts_per_period

        phi_arr = np.zeros(Noutputs)

        if return_angs:
            l_outer_arr = np.zeros(len(times))
            l_inner_arr = np.zeros(len(times))
            pom_test_arr = np.zeros(len(times))

        for i,t in enumerate(times): #Perform integration
            sim.integrate(t,exact_finish_time = exact)
            phi_arr[i] = (p*outer.l - q*inner.l - (p-q)*test.pomega)%(2*np.pi)

            if return_angs:
                l_outer_arr[i] = outer.l
                l_inner_arr[i] = inner.l
                pom_test_arr[i] = test.pomega


        if plot:
            markersize=1
            if dots:
                l_sty=' '
                marker = 'o'
            else:
                l_sty ='-'
                marker = ' '
            plt.figure(1,figsize=(11,6))

            if log_t:
                plt.semilogx(times/365.25,phi_arr,linestyle=l_sty,marker=marker,markersize=markersize)
            else:
                plt.plot(times/365.25,phi_arr,linestyle=l_sty,marker=marker,markersize=markersize)

            plt.xlabel("Time [Years]", fontsize=20)
            plt.ylabel(r"$\phi$ [rad]", fontsize=20)

            plt.tick_params(axis='both', which='major', labelsize=20)
            plt.tick_params(axis='both', which='minor', labelsize=8)

        if return_angs:
            return times,phi_arr,l_outer_arr,l_inner_arr,pom_test_arr
        else:
            return times, phi_arr

    def sort_data(self):

        """
        Sort the data in RV_files into one array in order of increasing time
        """

        assert len(self.RV_files) != 0 or len(self.JDs) !=0, "No data stored and no files to load"
        assert not(type(self.JDs[0]) is np.float64), "Unable to sort data"
        if len(self.JDs) == 0:
            self.load_data()
        if len(self.planets) != 0:
            assert len(self.vels_model) != 0, "No model velocities loaded, run self.log_like first"

        JDs = []
        vels = []
        errs = []
        vels_model = []

        for i in range(len(self.RV_files)):
            JDs = np.concatenate((JDs,self.JDs[i]))
            vels = np.concatenate((vels,(self.vels[i]-self.offsets[i])))
            errs = np.concatenate((errs,self.errs[i]))
            vels_model_tmp = self.vels_model[i] if len(self.vels_model) else np.zeros(len(self.JDs[i]))
            vels_model=np.concatenate((vels_model,vels_model_tmp))
        sort_arr = [JDs,vels,errs,vels_model]
        sort_arr = np.transpose(sort_arr)
        self.RV_data = sort_arr[np.argsort(sort_arr[:,0])]

    def periodogram(self,pnts=int(1e4),plot_per=True,show_perod=True,show_window=True,show_res=True,plot_range=0,ret=False):
        """
        Plot periodograms to analyze RV data.

        Parameters
        ----------

        pnts
            Number of periods or frequencies to use in calculating the power.
        plot_per: bool
            If true, make plot in terms of log period. Otherwise plot will be in terms of frequency=1/period
        show_perod: bool
            Plot the periodogram
        show_window: bool
            Plot the window function, which is the Fourier transform of the time sampling of the data.
        show_res: bool
            Plot a periodogram of the residuals to the data
        plot_range
            If False, uses default plotting range. Otherwise, plot_range should be a 1x2 array corresponding to the minimum and maximum plotted values of period or frequency.
        ret: bool
            If true, returns an array of the periods or frequencies, and then of the powers and phases used to make any of the plotted periodograms.
        """

        if not(type(self.JDs[0]) is np.float64):
            self.sort_data()
            times = self.RV_data[:,0]
            obs = self.RV_data[:,1]
            errs = self.RV_data[:,2]
            model = self.RV_data[:,3]
        else:
            assert len(self.vels_model) != 0 or not(show_res), "No model velocities loaded, run self.log_like first"
            times = self.JDs
            obs = self.vels
            errs = self.errs
            model = self.vels_model



        if not(plot_range): #Flag to specify plotted range as an input, otherwise defaults are below
            plot_range = [-0.3,4] if plot_per else [1e-3,4.0]
        if plot_per: #Flag to plot period instead of frequency
            periods = np.logspace(plot_range[0],plot_range[1],num=int(pnts))
            freqs = periods**(-1.)
            log_per = np.log10(periods)
            x_data = log_per
            x_lab = "Period [days]"
        else:
            freqs = np.linspace(plot_range[0],plot_range[1],num=int(pnts))
            x_data = freqs
            x_lab = 'Freq [1/day]'



        if ret:
            ret_arr = [x_data]

        def plotter(i,powers):
            fig = plt.figure(i,figsize=(6,6))

            plt.plot(x_data,powers)
            plt.axis([min(x_data),max(x_data),0,max(powers)*1.2])
            plt.xlabel(x_lab,fontsize=18)
            plt.ylabel('Power',fontsize=18)
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.tick_params(axis='both', which='minor', labelsize=8)
            plt.ylim(0,1)
            if plot_per:
                plt.xticks([0, 1, 2, 3, 4], [r'10$^{0}$',r'10$^{1}$',r'10$^{2}$',r'10$^{3}$',r'10$^{4}$'])
            return plt.gca()

        #Make the LS periodogram
        if show_perod:
            powers,phases = ls_period(freqs,times,obs,errs) #LS periodogram for the data. Formulas are from Zechmeister and Kurster (2009)
            ax = plotter(1,powers)
            ax.set_title("Periodogram", fontsize=20)

            if ret:
                ret_arr.append([powers,phases])


        if show_window:
            #Calculate the window function. See Dawson and Fabrycky (2010). Currently done in a non-FFT way, there's probably
            #a faster way to do it
            window = np.zeros(len(freqs),dtype=np.complex)
            N = len(times)

            for i,f in enumerate(freqs):
                # f = 1./p
                w = 0
                for t in times:
                    w+=np.exp(-2*np.pi*1j*f*t)
                window[i] = w/N

            win_powers = np.abs(window)
            # win_phases = np.arctan2(window.imag,window.real)
            win_phases = np.angle(window)

            ax = plotter(2,win_powers)
            ax.set_title("Window Function", fontsize=20)

            if ret:
                ret_arr.append([win_powers,win_phases])

        if show_res:
            #Periodogram of the residuals
            res = obs - model
            res_powers,res_phases = ls_period(freqs,times,res,errs)

            ax = plotter(3,res_powers)
            ax.set_title("Periodogram of Residuals", fontsize=20)
            if ret:
                ret_arr.append([res_powers,res_phases])

        if ret:
            return ret_arr

def ls_sums(f,times,obs,w_i):
    """Intermediate function for calculating the sums in the LS periodogram. See Zechmeister and Kurster (2009) for
    details"""

    omega = 2*np.pi*f
    C = np.sum(w_i*np.cos(omega*times))
    S = np.sum(w_i*np.sin(omega*times))
    YC_hat = np.sum(w_i*obs*np.cos(omega*times))
    YS_hat = np.sum(w_i*obs*np.sin(omega*times))
    CC_hat = np.sum(w_i*np.cos(omega*times)**2.)
    SS_hat = np.sum(w_i*np.sin(omega*times)**2.)
    CS_hat = np.sum(w_i*np.cos(omega*times)*np.sin(omega*times))

    return [C,S,YC_hat,YS_hat,CC_hat,SS_hat,CS_hat]

def ls_period(freqs,times,obs,errs):
    """Return the LS periodogram for a given set of data, given as (times,obs), with errors errs. The power is calculated
    at the frequencies that are given to the function. Uses a 'floating mean,' and weights the data points by the square
    of the error bars. See Zechmeister and Kurster (2009) for the formulas as well as a more in depth explanation"""

    W = np.sum(1/errs**2.)
    w_i = 1./(W*errs**2.)
    Y = np.sum(w_i*obs)
    YY_hat = np.sum(w_i*obs*obs)
    YY = YY_hat - Y*Y

    powers = np.zeros(len(freqs))
    phases = np.zeros(len(freqs))

    for i,f in enumerate(freqs):
        [C,S,YC_hat,YS_hat,CC_hat,SS_hat,CS_hat] = ls_sums(f,times,obs,w_i)
        YC = YC_hat - Y*C
        YS = YS_hat - Y*S
        CC = CC_hat - C*C
        SS = SS_hat - S*S
        CS = CS_hat - C*S
        D = CC*SS-CS**2.

        a = (YC*SS - YS*CS)/D
        b = (YS*CC - YC*CS)/D
        phases[i] = np.arctan2(b,a)
        powers[i] = (YY*D)**(-1.)*(SS*YC**2. + CC*YS**2. - 2*CS*YC*YS)

    return powers,phases


def kep_sol(t=0,t_p=0,e=0.1,per=100.):
    """
    Solve Kepler's equation to find the true anomaly of a planet at a given time.
    """
    n = 2*np.pi/per
    M = n*(t-t_p)

    kep = lambda E: M - E + e*np.sin(E)

    E = op.fsolve(kep,M)

    return 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))%(2*np.pi)

def RV_amp(m_star = 1.0, m_p = 9.5458e-4, omega = 0., i = np.pi/2., per = 365.25, f = 0., e = 0.0):

    """Calculate the radial velocity amplitude due to a planet for a given true anomaly"""

    deg2rad = np.pi/180.
    omega = omega*deg2rad
    G = 6.67259e-11
    m_sun = 1.988435e30
    JD_sec = 86400.0

    m_star_mks = m_star*m_sun
    m_p_mks = m_p*m_sun

    per_mks = per*JD_sec
    n = 2.*np.pi/per_mks
    a = (G*(m_star_mks + m_p_mks)/n**2.)**(1./3.)

    return np.sqrt(G/(m_star_mks + m_p_mks)/a/(1-e**2.))*(m_p_mks*np.sin(i))*(np.cos(omega+f)+e*np.cos(omega))