# *Bernoulli Mixture Models (BMM)*

This repository provides tools in a Python package *bernmix* for the unsupervised analysis of multivariate Bernoulli data with known number of cluster/groups using BMMs. 

## Maximum likelihood estimation 

Shows how to fit the model using [Expectation-Maximizition (EM)](https://github.com/AVoss84/bmm_mix/blob/master/EM_for_BMM.ipynb) algorithm as outlined in *Bishop (2006): Pattern Recognition and Machine Learning*. 

## Fully Bayesian estimation 

Shows how to fit the model using [Gibbs sampling](https://github.com/AVoss84/bmm_mix/blob/master/Gibbs_for_BMM.ipynb) algorithm.

```
from bernmix.utils import bmm_utils as bmm
```

### Installing

```
pip install -r requirements.txt
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Alexander Vosseler**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

