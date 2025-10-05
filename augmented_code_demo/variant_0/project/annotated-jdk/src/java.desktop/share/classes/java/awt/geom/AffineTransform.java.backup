/*
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package java.awt.geom;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.awt.Shape;
import java.beans.ConstructorProperties;
import java.io.IOException;
import java.io.Serial;

public class AffineTransform implements Cloneable, java.io.Serializable {

    public static final int TYPE_IDENTITY;

    public static final int TYPE_TRANSLATION;

    public static final int TYPE_UNIFORM_SCALE;

    public static final int TYPE_GENERAL_SCALE;

    public static final int TYPE_MASK_SCALE;

    public static final int TYPE_FLIP;

    public static final int TYPE_QUADRANT_ROTATION;

    public static final int TYPE_GENERAL_ROTATION;

    public static final int TYPE_MASK_ROTATION;

    public static final int TYPE_GENERAL_TRANSFORM;

    public AffineTransform() {
    }

    public AffineTransform(AffineTransform Tx) {
    }

    @ConstructorProperties({ "scaleX", "shearY", "shearX", "scaleY", "translateX", "translateY" })
    public AffineTransform(float m00, float m10, float m01, float m11, float m02, float m12) {
    }

    public AffineTransform(float[] flatmatrix) {
    }

    public AffineTransform(double m00, double m10, double m01, double m11, double m02, double m12) {
    }

    public AffineTransform(double[] flatmatrix) {
    }

    public static AffineTransform getTranslateInstance(double tx, double ty);

    public static AffineTransform getRotateInstance(double theta);

    public static AffineTransform getRotateInstance(double theta, double anchorx, double anchory);

    public static AffineTransform getRotateInstance(double vecx, double vecy);

    public static AffineTransform getRotateInstance(double vecx, double vecy, double anchorx, double anchory);

    public static AffineTransform getQuadrantRotateInstance(int numquadrants);

    public static AffineTransform getQuadrantRotateInstance(int numquadrants, double anchorx, double anchory);

    public static AffineTransform getScaleInstance(double sx, double sy);

    public static AffineTransform getShearInstance(double shx, double shy);

    public int getType();

    @SuppressWarnings("fallthrough")
    public double getDeterminant();

    void updateState();

    public void getMatrix(double[] flatmatrix);

    public double getScaleX();

    public double getScaleY();

    public double getShearX();

    public double getShearY();

    public double getTranslateX();

    public double getTranslateY();

    public void translate(double tx, double ty);

    public void rotate(double theta);

    public void rotate(double theta, double anchorx, double anchory);

    public void rotate(double vecx, double vecy);

    public void rotate(double vecx, double vecy, double anchorx, double anchory);

    public void quadrantRotate(int numquadrants);

    public void quadrantRotate(int numquadrants, double anchorx, double anchory);

    @SuppressWarnings("fallthrough")
    public void scale(double sx, double sy);

    public void shear(double shx, double shy);

    public void setToIdentity();

    public void setToTranslation(double tx, double ty);

    public void setToRotation(double theta);

    public void setToRotation(double theta, double anchorx, double anchory);

    public void setToRotation(double vecx, double vecy);

    public void setToRotation(double vecx, double vecy, double anchorx, double anchory);

    public void setToQuadrantRotation(int numquadrants);

    public void setToQuadrantRotation(int numquadrants, double anchorx, double anchory);

    public void setToScale(double sx, double sy);

    public void setToShear(double shx, double shy);

    public void setTransform(AffineTransform Tx);

    public void setTransform(double m00, double m10, double m01, double m11, double m02, double m12);

    @SuppressWarnings("fallthrough")
    public void concatenate(AffineTransform Tx);

    @SuppressWarnings("fallthrough")
    public void preConcatenate(AffineTransform Tx);

    public AffineTransform createInverse() throws NoninvertibleTransformException;

    public void invert() throws NoninvertibleTransformException;

    public Point2D transform(Point2D ptSrc, Point2D ptDst);

    public void transform(Point2D[] ptSrc, int srcOff, Point2D[] ptDst, int dstOff, int numPts);

    public void transform(float[] srcPts, int srcOff, float[] dstPts, int dstOff, int numPts);

    public void transform(double[] srcPts, int srcOff, double[] dstPts, int dstOff, int numPts);

    public void transform(float[] srcPts, int srcOff, double[] dstPts, int dstOff, int numPts);

    public void transform(double[] srcPts, int srcOff, float[] dstPts, int dstOff, int numPts);

    @SuppressWarnings("fallthrough")
    public Point2D inverseTransform(Point2D ptSrc, Point2D ptDst) throws NoninvertibleTransformException;

    public void inverseTransform(double[] srcPts, int srcOff, double[] dstPts, int dstOff, int numPts) throws NoninvertibleTransformException;

    public Point2D deltaTransform(Point2D ptSrc, Point2D ptDst);

    public void deltaTransform(double[] srcPts, int srcOff, double[] dstPts, int dstOff, int numPts);

    public Shape createTransformedShape(Shape pSrc);

    public String toString();

    public boolean isIdentity();

    public Object clone();

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);
}
