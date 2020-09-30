# NNet
This library was created to facilitate the process of creating and manipulating neural networks. This library will be of interest to those who are just starting their way in the study of neural networks and, most likely, will not be suitable for use in large projects. However, if you made it work in a serious project, congratulations.
## How to build it
To turn this obscure code into a ready-made library, go to the cloned folder and use the dotnet build command.

    dotnet build ./src/NNet.csproj
## How to use it
To start working with this library, you need to add a link to it in your project. It is worth noting that this library is written in version .Net Core 3.1, so make sure that your project can work with it.

After that, to create a neural network, you need to use the builder, through which you will create your neural network.

    var builder = NeuralNetwork.Builder;

The further creation process consists of calling the construction methods. Please note that the builder returns an intermediate result, rather than a ready-made network, when the creation process is complete. To convert it to a ready-made network, call the conversion method.

    var network = builderResult.ToNetwork();
## License
This library is available under the [MIT LIcense](https://github.com/JustBadCoder/NNet/blob/master/LICENSE)