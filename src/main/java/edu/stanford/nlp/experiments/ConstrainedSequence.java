package edu.stanford.nlp.experiments;

import gurobi.*;

/**
 * Created by keenon on 2/4/15.
 *
 * Hopefully will solve a sequence labeling problem where we constrain the labels to take on a set of values no more
 * than a certain count per label.
 */
public class ConstrainedSequence {

    public static void main(String[] args) {
        int[] classes = solve(new double[][]{
                new double[]{1.0, 4.0},
                new double[]{2.0, 3.5}
        }, new int[]{1, 1}, new int[]{-1, -1});

        System.out.println(classes[0]+","+classes[1]);
    }

    public static int[] solve(double[][] probabilities, int[] allowedClassOccupants, int[] forcedClasses) {
        assert(probabilities[0].length == allowedClassOccupants.length);

        int[] classes = new int[probabilities.length];

        try {
            GRBEnv env = new GRBEnv();
            GRBModel model = new GRBModel(env);

            GRBVar[][] vars = new GRBVar[probabilities.length][];
            for (int i = 0; i < probabilities.length; i++) {
                vars[i] = new GRBVar[probabilities[i].length];
                for (int j = 0; j < probabilities[i].length; j++) {
                    vars[i][j] = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "x"+i+j);

                    probabilities[i][j] *= -1;
                }
            }
            model.update();

            // Make sure all columns sum to 1, i.e. all values get an assignment

            for (int i = 0; i < probabilities.length; i++) {
                GRBLinExpr expr = new GRBLinExpr();
                for (int j = 0; j < vars[i].length; j++) {
                    expr.addTerm(1.0, vars[i][j]);
                }
                model.addConstr(expr, GRB.EQUAL, 1.0, "c"+i);
            }

            // Make sure that all the classes have no more than the allowed number of occupants

            for (int j = 0; j < allowedClassOccupants.length; j++) {
                if (allowedClassOccupants[j] != -1) {
                    GRBLinExpr expr = new GRBLinExpr();
                    for (int i = 0; i < probabilities.length; i++) {
                        expr.addTerm(1.0, vars[i][j]);
                    }
                    model.addConstr(expr, GRB.LESS_EQUAL, allowedClassOccupants[j], "d" + j);
                }
            }

            // Constrain any forced classes

            for (int i = 0; i < forcedClasses.length; i++) {
                if (forcedClasses[i] != -1) {
                    for (int j = 0; j < probabilities[i].length; j++) {
                        GRBLinExpr expr = new GRBLinExpr();
                        expr.addTerm(1.0, vars[i][j]);
                        if (forcedClasses[i] == j) {
                            model.addConstr(expr, GRB.EQUAL, 1.0, "f"+i+j);
                        }
                        else {
                            model.addConstr(expr, GRB.EQUAL, 0.0, "f"+i+j);
                        }
                    }
                }
            }

            // Add the goal

            GRBLinExpr expr = new GRBLinExpr();
            for (int i = 0; i < probabilities.length; i++) {
                for (int j = 0; j < vars[i].length; j++) {
                    expr.addTerm(probabilities[i][j], vars[i][j]);
                }
            }
            model.setObjective(expr);

            model.optimize();

            // Get the values of the model

            for (int i = 0; i < probabilities.length; i++) {
                for (int j = 0; j < vars[i].length; j++) {
                    if (vars[i][j].get(GRB.DoubleAttr.X) == 1) {
                        classes[i] = j;
                        break;
                    }
                }
            }

            // Dispose of model and environment
            model.dispose();
            env.dispose();

        } catch (GRBException e) {
            System.out.println("Error code: " + e.getErrorCode() + ". " +
                    e.getMessage());
        }

        return classes;
    }
}
