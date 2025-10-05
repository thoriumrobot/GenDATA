/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package java.awt.geom;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.awt.Shape;
    @Positive
import java.beans.ConstructorProperties;
    @Positive
import java.io.IOException;
    @Positive
import java.io.Serial;

    @Positive
public class AffineTransform implements Cloneable, java.io.Serializable {

    @Positive
    public static final int TYPE_IDENTITY;

    @Positive
    public static final int TYPE_TRANSLATION;

    @Positive
    public static final int TYPE_UNIFORM_SCALE;

    @Positive
    public static final int TYPE_GENERAL_SCALE;

    @Positive
    public static final int TYPE_MASK_SCALE;

    @Positive
    public static final int TYPE_FLIP;

    @Positive
    public static final int TYPE_QUADRANT_ROTATION;

    @Positive
    public static final int TYPE_GENERAL_ROTATION;

    @Positive
    public static final int TYPE_MASK_ROTATION;

    @Positive
    public static final int TYPE_GENERAL_TRANSFORM;

    @Positive
    public AffineTransform() {
    @Positive
    }

    @Positive
    public AffineTransform(AffineTransform Tx) {
    @Positive
    }

    @Positive
    @ConstructorProperties({ "scaleX", "shearY", "shearX", "scaleY", "translateX", "translateY" })
    @Positive
    public AffineTransform(float m00, float m10, float m01, float m11, float m02, float m12) {
    @Positive
    }

    @Positive
    public AffineTransform(float[] flatmatrix) {
    @Positive
    }

    @Positive
    public AffineTransform(double m00, double m10, double m01, double m11, double m02, double m12) {
    @Positive
    }

    @Positive
    public AffineTransform(double[] flatmatrix) {
    @Positive
    }

    @Positive
    public static AffineTransform getTranslateInstance(double tx, double ty);

    @Positive
    public static AffineTransform getRotateInstance(double theta);

    @Positive
    public static AffineTransform getRotateInstance(double theta, double anchorx, double anchory);

    @Positive
    public static AffineTransform getRotateInstance(double vecx, double vecy);

    @Positive
    public static AffineTransform getRotateInstance(double vecx, double vecy, double anchorx, double anchory);

    @Positive
    public static AffineTransform getQuadrantRotateInstance(int numquadrants);

    @Positive
    public static AffineTransform getQuadrantRotateInstance(int numquadrants, double anchorx, double anchory);

    @Positive
    public static AffineTransform getScaleInstance(double sx, double sy);

    @Positive
    public static AffineTransform getShearInstance(double shx, double shy);

    @Positive
    public int getType();

    @Positive
    @SuppressWarnings("fallthrough")
    @Positive
    public double getDeterminant();

    @Positive
    void updateState();

    @Positive
    public void getMatrix(double[] flatmatrix);

    @Positive
    public double getScaleX();

    @Positive
    public double getScaleY();

    @Positive
    public double getShearX();

    @Positive
    public double getShearY();

    @Positive
    public double getTranslateX();

    @Positive
    public double getTranslateY();

    @Positive
    public void translate(double tx, double ty);

    @Positive
    public void rotate(double theta);

    @Positive
    public void rotate(double theta, double anchorx, double anchory);

    @Positive
    public void rotate(double vecx, double vecy);

    @Positive
    public void rotate(double vecx, double vecy, double anchorx, double anchory);

    @Positive
    public void quadrantRotate(int numquadrants);

    @Positive
    public void quadrantRotate(int numquadrants, double anchorx, double anchory);

    @Positive
    @SuppressWarnings("fallthrough")
    @Positive
    public void scale(double sx, double sy);

    @Positive
    public void shear(double shx, double shy);

    @Positive
    public void setToIdentity();

    @Positive
    public void setToTranslation(double tx, double ty);

    @Positive
    public void setToRotation(double theta);

    @Positive
    public void setToRotation(double theta, double anchorx, double anchory);

    @Positive
    public void setToRotation(double vecx, double vecy);

    @Positive
    public void setToRotation(double vecx, double vecy, double anchorx, double anchory);

    @Positive
    public void setToQuadrantRotation(int numquadrants);

    @Positive
    public void setToQuadrantRotation(int numquadrants, double anchorx, double anchory);

    @Positive
    public void setToScale(double sx, double sy);

    @Positive
    public void setToShear(double shx, double shy);

    @Positive
    public void setTransform(AffineTransform Tx);

    @Positive
    public void setTransform(double m00, double m10, double m01, double m11, double m02, double m12);

    @Positive
    @SuppressWarnings("fallthrough")
    @Positive
    public void concatenate(AffineTransform Tx);

    @Positive
    @SuppressWarnings("fallthrough")
    @Positive
    public void preConcatenate(AffineTransform Tx);

    @Positive
    public AffineTransform createInverse() throws NoninvertibleTransformException;

    @Positive
    public void invert() throws NoninvertibleTransformException;

    @Positive
    public Point2D transform(Point2D ptSrc, Point2D ptDst);

    @Positive
    public void transform(Point2D[] ptSrc, int srcOff, Point2D[] ptDst, int dstOff, int numPts);

    @Positive
    public void transform(float[] srcPts, int srcOff, float[] dstPts, int dstOff, int numPts);

    @Positive
    public void transform(double[] srcPts, int srcOff, double[] dstPts, int dstOff, int numPts);

    @Positive
    public void transform(float[] srcPts, int srcOff, double[] dstPts, int dstOff, int numPts);

    @Positive
    public void transform(double[] srcPts, int srcOff, float[] dstPts, int dstOff, int numPts);

    @Positive
    @SuppressWarnings("fallthrough")
    @Positive
    public Point2D inverseTransform(Point2D ptSrc, Point2D ptDst) throws NoninvertibleTransformException;

    @Positive
    public void inverseTransform(double[] srcPts, int srcOff, double[] dstPts, int dstOff, int numPts) throws NoninvertibleTransformException;

    @Positive
    public Point2D deltaTransform(Point2D ptSrc, Point2D ptDst);

    @Positive
    public void deltaTransform(double[] srcPts, int srcOff, double[] dstPts, int dstOff, int numPts);

    @Positive
    public Shape createTransformedShape(Shape pSrc);

    @Positive
    public String toString();

    @Positive
    public boolean isIdentity();

    @Positive
    public Object clone();

    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);
    @Positive
}
