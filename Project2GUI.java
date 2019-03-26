/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
//package project2gui;

import java.util.Arrays;
import javafx.geometry.Insets;
import javafx.application.Application;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.ToggleButton;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.FlowPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.scene.text.Text;
import javafx.stage.Stage;
import java.util.*;
import java.text.*;
import java.io.PrintWriter;
import java.io.File;
import java.io.IOException;
//import NeuralNetwork.java;
import java.util.*;
import java.lang.Math.*;
import static java.lang.Math.abs;
import static java.lang.Math.exp;
import static java.lang.Math.pow;




/**
 *
 * @author Preston-PC

 */

 class NeuralNetwork {
	 
    double[][] w1 = new double[35][32];				//weights between input and l1
	double[][] z2 = new double[26][32];				//layer 1 before activation
	double[][] a2 = new double[26][32];				//layer 1 after activation
	double[][] w2 = new double[32][26];				//weights between l1 and output
	double[][] z3 = new double[26][26];				//output before activation
	double[][] a3 = new double[26][32];				//layer 2 after activation
	double[][] yhat = new double[26][26];			//guestimation output from forward
	
	double[][] out_delta = new double[26][26];		//output delta for back propagation
	double[][] l1_delta = new double[26][32];		//layer 1 delta for back propagation
	
	double alpha = 0.07;							//learning rate alpha
	
	
	//Training Input Data
	double [][] input = {{0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1}, 					
						{1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0},				 
						{0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0},				 
						{1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0},
						{1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1},
						{1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0},
						{0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1},
						{1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1},
						{0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0},
						{0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0},
						{1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1},
						{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1},
						{1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1},
						{1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1},
						{0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0},
						{1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0},
						{0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1},
						{1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1},
						{0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0},
						{1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0},
						{1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0},
						{1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0},
						{1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
						{1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1},
						{1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0},
						{1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1}};
	
				   
	//Training output Data
	double[][] output = {{1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},	
						{0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
						{0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},	
						{0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
						{0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},	
						{0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
						{0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},	
						{0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
						{0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},	
						{0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
						{0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},	
						{0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
						{0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
						{0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0},	
						{0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0},
						{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},	
						{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0},
						{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0},	
						{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0},
						{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0},	
						{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0},
						{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0},	
						{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0},
						{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0},	
						{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0},
						{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1}};
	
	
	//MATRICIES FOR POST TRAINED NETWORK
	double[][] post_input = new double[1][35];
	double[][] post_output = new double[1][26];
	
	
	//constructor to set up neural network
	public NeuralNetwork(){
		populate(w1);
		populate(w2);
	}
	
	//This is the forward propagation for training
	public void forward(){
		
		//gets the layer 1 values before activation
		z2 = dotp(input,w1);
		
		//applies activation to layer 1
		a2 = tan_activation(z2);
		
		//gets the output layer values before activation
		z3 = dotp(a2,w2);
		
		//applies activation to output layer
		yhat = sig_activation(z3);
	}
	
	//This is the backward propagation for training
	public void backward(){
		
		//calculating the deltas for output layer and hidden layer
		out_delta = matrix_mult(sig_deriv(z3),matrix_subtr(output,yhat));
		l1_delta = matrix_mult(tan_deriv(z2),dotp(out_delta,transpose(w2)));
		
		//updating weights for w2
		for(int k = 0; k < z2.length; k++){
			for(int i = 0; i < w2.length; i++){
				for(int j = 0; j < w2[0].length; j++){
					w2[i][j] += alpha * z2[k][i]* out_delta[k][j];
				}
			}
		}
		
		
		//updating weights for w1
		for(int k = 0; k < input.length;k++){
			for(int i = 0; i < w1.length; i++){
				for(int j = 0; j < w1[0].length; j++){
					w1[i][j] += alpha * input[k][i] * l1_delta[k][j];
				}
			}
		}
	}
	
	//function to train the neural net. Forward and backward.
	public void learn(){
		System.out.println("\n\nNeural Network is Learning.\n\nThis should take roughly 10 seconds...\n");
		 
	 	//Trains the network
		for(int i = 0; i < 400; i++){
			for(int j = 0; j<10;j++){
				forward();
				backward();
			}	
			System.out.println("Accuracy for iteration " + i + " is : " + accuracy() + " \n\n");	
		}
	}
	
	//forward propagation for post-trained network
	public double[][] post_trained_forward(double[] in){
		
		//converts the input array into a matrix
		clone(in,post_input);
		
		//gets the layer 1 values before activation
		z2 = dotp(post_input,w1);
		
		//applies activation to layer 1
		a2 = tan_activation(z2);
		
		//gets the output layer values before activation
		z3 = dotp(a2,w2);
		
		//applies activation to output layer
        post_output = sig_activation(z3);
        
        return post_output;
	}
	
	//function to calculate noise tolerance
	public void noise(double[] in, int num, String let){
		
		double[] temp = in;		//allows input to be modified
		boolean[] checked = new boolean[35];
		double highest = 0.0; //keeps track of highest output
		int index = -1;	 //keeps track of index of highest output
		int tab = 0; 	//keeps track of tabulations
		Random r = new Random();	//allows random selection of bit to be flipped
		int flip = 0;
		
		//flips a bit at random
		for(int i = 0; i<35; i++){
			flip = r.nextInt(34);
			highest = 0;
			index = -1;
			
			if(!checked[flip]){
				if(temp[flip] == 0){
					temp[flip] = 1;
					checked[flip] = true;
				}
				else if(temp[flip] == 1){
					temp[flip] = 0;
					checked[flip] = true;
				}
			}
			
			//propagates the modified data
			post_trained_forward(temp);
			
				//finds the highest output for this propagation
				for(int k = 0; k < post_output[0].length;k++){
					if(highest < post_output[0][k]){
						highest = post_output[0][k];
						index = k;
					}
					
				}
				//increases if the network still recognizes the letter
				if(index == num){
					tab++;
				}
		}
		
		//prints the number of bits flipped before network failed to recognize letter
		System.out.println("# of tabulations for " + let + ": " + tab);
	}
	
	//BELOW ARE THE HELPER FUNCTIONS I MADE FOR THIS PROJECT
	
	//function to calculate how accurate the Network is after X iterations
	public double accuracy(){
		
		double sum = 0;
		double avg = 0;
		
		//sum the output accuracies
		for(int i = 0; i < yhat.length; i++){
				sum += yhat[i][i];
		}
		//find the average
		avg = sum/(yhat.length);
		
		return avg;
	}
	
	//helper function for element-wise multiplication
	private double[][] matrix_mult(double[][]in,double[][]w){
		double[][] out = new double[in.length][in[0].length];
		for(int i = 0; i < in.length;i++){
			for(int j = 0; j < in[i].length; j++){
				out[i][j] = in[i][j] * w[i][j];
			}
		}
		return out;
	}
	
	//helper function for matrix subtraction
	private double[][] matrix_subtr(double[][]in,double[][]w){
		double[][] out = new double[in.length][in[0].length];
		for(int i = 0; i < in.length;i++){
			for(int j = 0; j < in[i].length; j++){
				out[i][j] = in[i][j] - w[i][j];
			}
		}
		return out;
	}
	
	//helper function to transpose matricies
	private double[][] transpose(double[][]in){
		double[][] out = new double[in[0].length][in.length];
		for(int i = 0; i < in.length;i++){
			for(int j = 0; j < in[i].length; j++){
				out[j][i] = in[i][j];
			}
		}
		return out;
	}
		
	//tanh activiation function		
	private double[][] tan_activation(double[][] n){
		double[][] out = new double[n.length][n[0].length];
		for(int i = 0; i < n.length;i++){
			for(int j = 0; j < n[i].length;j++){
				out[i][j] = Math.sinh(n[i][j])/Math.cosh(n[i][j]);
			}
		}
		return out;
	}
	
	//tanh derivitive function
	private double[][] tan_deriv(double[][] n){
		double[][] out = new double[n.length][n[0].length];
		for(int i = 0; i < n.length;i++){
			for(int j = 0; j < n[i].length;j++){
				out[i][j] = 1-(Math.pow(Math.sinh(n[i][j]),2)/Math.pow(Math.cosh(n[i][j]),2));
			}
		}
		return out;
	}
	
	//sigmoid derivative function
	private double[][] sig_deriv(double[][] n){
		double[][] out = new double[n.length][n[0].length];
		for(int i = 0; i < n.length;i++){
			for(int j = 0; j < n[i].length;j++){
				out[i][j] = (1/(1 + Math.pow(Math.E, -(n[i][j]))))*(1-(1/(1 + Math.pow(Math.E, -(n[i][j])))));
			}
		}
		return out;
	}
	
	//sigmoid activation function
	private double[][] sig_activation(double[][] n){
		double[][] out = new double[n.length][n[0].length];
		for(int i = 0; i < n.length;i++){
			for(int j = 0; j < n[i].length;j++){
				out[i][j] = 1/(1 + Math.pow(Math.E, -(n[i][j])));
			}
		}
		return out;
	}
	
	//helper function to populate matricies
	private void populate(double[][]in){
		
		int coinflip;	//provides a 50/50 chance of being a 0 or 1
		Random rng = new Random();
		for(int i=0;i<in.length;i++){
			for(int j=0;j<in[i].length;j++){
				coinflip = rng.nextInt(100);
				if(coinflip >50){
					in[i][j] = 0.1;
				}
				else{
					in[i][j] = -0.1;
				}
				
			}
		}
	}
	
	//helper function to convert arrays to matricies
	private void clone(double[]in, double[][] out){
		for(int i = 0; i < in.length; i++){
			out[0][i] = in[i];
		}
	}
	
	//helper function for dot product
	private double[][] dotp(double[][]in, double[][]w){
		double[][] out = new double[in.length][w[0].length];
		for(int i = 0; i < in.length;i++){
			for(int j = 0; j < w[0].length; j++){
				for(int k = 0; k < in[0].length;k++){
					out[i][j] += (in[i][k]*w[k][j]);
				}
			}
		}
		return out;
	}
	
	//helper function to print matrix
	public void printmatrix(double[][]in){
		for(int i=0;i<in.length;i++){
			System.out.print("[");
			for(int j=0;j<in[i].length;j++){
				if(j!=in[i].length){
					System.out.print(in[i][j] + ",");
				}
				else{
					System.out.print(in[i][j]);
				}
			}
			System.out.print("]\n");
		}
		System.out.println("\n\n");
	}
	
	//prints the output for letter 
	public void printoutput(String let){
		double highest = 0;
		double num = 0;
		
		for(int i = 0; i < post_output[0].length; i++){
			if(post_output[0][i] > highest){
				highest = post_output[0][i];
				num = i;
			}
		}
		System.out.println("Letter Guessed: " + String.valueOf((char)(num + 65)) + ". Highest output was :" + highest + ".\n");
	}
	
}
 



































public class Project2GUI extends Application {
    
    //do not use these variables anywhere unless you are Preston and/or ask him
    int globalincr = 0;
    int butval = 0;
    private double[] globalinput = new double[35]; //array for togglebutton input to be sent to neural net
    private double[] globaloutput = new double[10]; //array for neural net results, 0-4 being percent, 5-9 being letter
    ToggleButton gridbutton[][] = new ToggleButton[7][5]; //Array of array of togglebuttons
    Button update = new Button("Update Results"); //initialized update button in global scope so evaluate button can trigger it to update

    
    public double[] AthenaRunner(double[] in){
        in = getMyInput(); //gets input from GUI
        
        //////////////////////////////////////////////////////////NNRunner
        //System.out.println("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"); //clears the screen for the next run
		
		int sample = 26;		//set the number of letters given to the NN
                
		double[][] out;	//the output matrix from forward
		double[] output = {0,0,0,0,0,0,0,0,0,0};	//output array that Preston needs. still need to code that algorithm
		
		//Instanciate the NN
        NeuralNetwork athena = new NeuralNetwork();
        athena.learn();

		//sets output matrix to result from forward propagation
		out = athena.post_trained_forward(in);
		
        ////////////////////////////////////////////////////////////////////////
		
		for(int j = 0; j < 5; j++){
			double highest = 0;
			int num = 0;
			
			for(int i = 0; i < out[0].length; i++){
					if(out[0][i] > highest){
						highest = out[0][i];
						num = i;
					}
					 //GETS RID OF HIGHEST SO WE CAN GET NEXT HIGHEST
				}
			
			out[0][num] = 0;

			output[j] = highest; //ELEMENT
			output[j+5] = num; //INDEX
			
			
		}

                
        return output; //returns output to GUI
    }
    
    //This holds the grid value to be passed to neural net
    public double[] getMyInput(){
        return this.globalinput;
    }
    
    //setter function
    public void setMyInput(double[] values){
        this.globalinput = values;
    }
    
    //This holds the result from neural net
    public double[] getMyResult(){
        return this.globaloutput;
    }
    
    //setter function
    public void setMyResult(double[] values){
        this.globaloutput = values;
    }
    
    //converts number 0-25 to letter
    private String num2letter(double i) {
    int j = (int) i;
    char[] alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".toCharArray();
    if (j > 25) {
            return null;
        }
        return Character.toString(alphabet[j]);
    }
    
    //function to check grid buttons
    public int buttonpress( int x, int y){
            if(gridbutton[y][x].isSelected() == true){
                butval = 1;
                //gridbutton[y][x].setStyle("-fx-background-color: #000000;");
                
            }
            else if(gridbutton[y][x].isSelected() == false){
                butval = 0;
                //gridbutton[y][x].setStyle("-fx-background-color: #FFFFFF;");
            }
       return butval;
    } 
    
    //function to reset button grid
    public void buttonreset(int x, int y){
        gridbutton[y][x].setSelected(false);
    }   
    
    //converts 35 member array to 5x7 matrix
    public double[][] array2matrix(double[] arr){
        int index = 0;
        double[][] matr = new double [5][7];
        for(int y = 0; y < 7; ++y){
            for(int x = 0; x < 5; ++x){
                matr[x][y] = arr[index];
                ++index;
            }
        }
        return matr;
    }
    
    //converts 5x7 matrix to 35 member array
    public double[] matrix2array(double[][] matr){
        int index = 0;
        double[] arr = new double[35];
        for(int y = 0; y < 7; ++y){
            for(int x = 0; x < 5; ++x){
                arr[index] = matr[x][y];
                ++index;
            }
        }
        return arr;
    }
    
    
    
    @Override
    public void start(Stage primaryStage) {
        //Title
        primaryStage.setTitle("Group 6 Neural Net: Athena");
        //Initialized main border pane and then adds sections
        BorderPane border = new BorderPane();
        border.setCenter(addGridPane());
        border.setBottom(addHBox());
        border.setRight(addVBox());
        border.setTop(addtitle());
        border.setLeft(addFlowPane());
    
        Scene scene = new Scene(border);
        primaryStage.setScene(scene);
        primaryStage.show();
    }
    
    public GridPane addGridPane() { //adds toggle button grid
        
        int BUTTON_PADDING = 2;
        int NUM_BUTTON_LINES = 7;
        int BUTTONS_PER_LINE = 5;
        
        GridPane grid = new GridPane();
        grid.setPadding(new Insets(BUTTON_PADDING));
        grid.setHgap(BUTTON_PADDING);
        grid.setVgap(BUTTON_PADDING);
        
        for (int y = 0; y < NUM_BUTTON_LINES; y++) {
            for (int x = 0; x < BUTTONS_PER_LINE; x++) {
                //String strx = Integer.toString(y);
                //String stry = Integer.toString(x);
                //String coord = 'b' + strx + stry;
                gridbutton[y][x] = new ToggleButton();
                gridbutton[y][x].setMinWidth(40);
                gridbutton[y][x].setMinHeight(40);
                //gridbutton[y][x] = 
                gridbutton[y][x].getStylesheets().add(this.getClass().getResource("controlStyle1.css").toExternalForm());
                grid.add(gridbutton[y][x], x, y);
            }
        }
        return grid;
    }
    
    public HBox addHBox() { //sets actions of bottom buttons
        HBox hbox = new HBox();
        hbox.setPadding(new Insets(15, 10, 15, 210));
        hbox.setSpacing(10);   // Gap between nodes
        Button evalbtn = new Button("Evaluate");
        evalbtn.setPrefSize(100, 20);
        
        double[] localinput = new double[35];
        
        evalbtn.setOnAction(new EventHandler<ActionEvent>() {
                    @Override
                    public void handle(ActionEvent event) {
                        globalincr = 0;
                        for(int x = 0; x < 7; x++){
                            for(int y = 0; y < 5; y++){
                                localinput[globalincr] = buttonpress(y,x);
                                ++globalincr;
                            }
                        }
                        setMyInput(localinput);
                        System.out.println(Arrays.toString(getMyInput()));
                        setMyResult(AthenaRunner(getMyInput()));
                        
                        update.fire();
                    }
                });
       
        Button resetbtn = new Button("Reset");
        resetbtn.setPrefSize(100, 20);
        
        resetbtn.setOnAction(new EventHandler<ActionEvent>() {
                    @Override
                    public void handle(ActionEvent event) {
                        for(int x = 0; x < 5; x++){
                            for(int y = 0; y < 7; y++){
                                buttonreset(x,y);
                            }
                        }
                    }
                });
  
        hbox.getChildren().addAll(evalbtn, resetbtn);
        return hbox;

    }
    
    private VBox addVBox() { //Sets results
        
        VBox vbox = new VBox();
        vbox.setPadding(new Insets(10)); // Set all sides to 10
        vbox.setSpacing(8);              // Gap between nodes

        Text title = new Text("Results:        ");
        title.setFont(Font.font("Arial", FontWeight.BOLD, 14));
        vbox.getChildren().add(title);
        //Button update = new Button("Update Results");
        
        int[] resarr = new int[10];
        
        //test values
        resarr[0] = 0;
        resarr[1] = 0;
        resarr[2] = 0;
        resarr[3] = 0;
        resarr[4] = 0;
        resarr[5] = 0;
        resarr[6] = 0;
        resarr[7] = 0;
        resarr[8] = 0;
        resarr[9] = 0;
        
        //just examples below
        String a = num2letter(resarr[5]) + ": " + Double.toString(resarr[0]);
        String b = num2letter(resarr[6]) + ": " + Double.toString(resarr[1]);
        String c = num2letter(resarr[7]) + ": " + Double.toString(resarr[2]);
        String d = num2letter(resarr[8]) + ": " + Double.toString(resarr[3]);
        String e = num2letter(resarr[9]) + ": " + Double.toString(resarr[4]);
        
        Text results[] = new Text[] {
            new Text(a),
            new Text(b),
            new Text(c),
            new Text(d),
            new Text(e)};
			
		Text letteroutput = new Text();
        
       

        for (int i=0; i<5; i++) {
            // Add offset to left side to indent from title
            VBox.setMargin(results[i], new Insets(0, 0, 0, 8));
            vbox.getChildren().add(results[i]);
        }
        
        update.setOnAction(new EventHandler<ActionEvent>() {
                    @Override
                    public void handle(ActionEvent event) {
                       
                        double[] finalarray = new double[10];
                        
                        finalarray = getMyResult();
                        
                       String a = num2letter(finalarray[5]) + ": " + Double.toString(finalarray[0]);
                       String b = num2letter(finalarray[6]) + ": " + Double.toString(finalarray[1]);
                       String c = num2letter(finalarray[7]) + ": " + Double.toString(finalarray[2]);
                       String d = num2letter(finalarray[8]) + ": " + Double.toString(finalarray[3]);
                       String e = num2letter(finalarray[9]) + ": " + Double.toString(finalarray[4]);
                        
                       results[0].setText(a.substring(0,10));
                       results[1].setText(b.substring(0,10));
                       results[2].setText(c.substring(0,10));
                       results[3].setText(d.substring(0,10));
                       results[4].setText(e.substring(0,10));
					   
					   letteroutput.setText("Based on these\nresults our best\nguess for the\nletter is: " + num2letter(finalarray[5]) + ".");
                        
                    }
                });
        vbox.setStyle("-fx-background-color: DAE6F3;");
        vbox.getChildren().add(letteroutput);          
        return vbox;
    }
    
    private HBox addtitle(){ //sets title at top
        HBox hbox = new HBox();
        hbox.setPadding(new Insets(10,170,10,225)); 
        hbox.setSpacing(8);              
        Text title = new Text("Welcome to our Neural Net");
        title.setFont(Font.font("Arial", FontWeight.BOLD, 14));
        hbox.getChildren().add(title);
        return hbox;
    }
    
    public FlowPane addFlowPane() { //sets instruction box to left
        FlowPane flow = new FlowPane();
        flow.setPadding(new Insets(5, 0, 5, 0));
        flow.setVgap(4);
        flow.setHgap(4);
        flow.setPrefWrapLength(170); // preferred width allows for two columns
        flow.setStyle("-fx-background-color: DAE6F3;");
        
        Text description = new Text(" Instructions:\n Please click on the boxes in the 5x7\n grid in order to make them resemble a capital letter.\n Once you have done this please click evaluate"
                + "\n and our neural net Athena will identify the letter you made.\n ");
        
        flow.getChildren().add(description);

        return flow;
    }
}

  

