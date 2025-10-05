/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.io.Serial;
    @Positive
import java.io.Serializable;

    @Positive
public abstract class RoundRectangle2D extends RectangularShape {

    @Positive
    public static class Float extends RoundRectangle2D implements Serializable {

    @Positive
        public float x;

    @Positive
        public float y;

    @Positive
        public float width;

    @Positive
        public float height;

    @Positive
        public float arcwidth;

    @Positive
        public float archeight;

    @Positive
        public Float() {
    @Positive
        }

    @Positive
        public Float(float x, float y, float w, float h, float arcw, float arch) {
    @Positive
        }

    @Positive
        public double getX();

    @Positive
        public double getY();

    @Positive
        public double getWidth();

    @Positive
        public double getHeight();

    @Positive
        public double getArcWidth();

    @Positive
        public double getArcHeight();

    @Positive
        public boolean isEmpty();

    @Positive
        public void setRoundRect(float x, float y, float w, float h, float arcw, float arch);

    @Positive
        public void setRoundRect(double x, double y, double w, double h, double arcw, double arch);

    @Positive
        public void setRoundRect(RoundRectangle2D rr);

    @Positive
        public Rectangle2D getBounds2D();
    @Positive
    }

    @Positive
    public static class Double extends RoundRectangle2D implements Serializable {

    @Positive
        public double x;

    @Positive
        public double y;

    @Positive
        public double width;

    @Positive
        public double height;

    @Positive
        public double arcwidth;

    @Positive
        public double archeight;

    @Positive
        public Double() {
    @Positive
        }

    @Positive
        public Double(double x, double y, double w, double h, double arcw, double arch) {
    @Positive
        }

    @Positive
        public double getX();

    @Positive
        public double getY();

    @Positive
        public double getWidth();

    @Positive
        public double getHeight();

    @Positive
        public double getArcWidth();

    @Positive
        public double getArcHeight();

    @Positive
        public boolean isEmpty();

    @Positive
        public void setRoundRect(double x, double y, double w, double h, double arcw, double arch);

    @Positive
        public void setRoundRect(RoundRectangle2D rr);

    @Positive
        public Rectangle2D getBounds2D();
    @Positive
    }

    @Positive
    protected RoundRectangle2D() {
    @Positive
    }

    @Positive
    public abstract double getArcWidth();

    @Positive
    public abstract double getArcHeight();

    @Positive
    public abstract void setRoundRect(double x, double y, double w, double h, double arcWidth, double arcHeight);

    @Positive
    public void setRoundRect(RoundRectangle2D rr);

    @Positive
    public void setFrame(double x, double y, double w, double h);

    @Positive
    public boolean contains(double x, double y);

    @Positive
    public boolean intersects(double x, double y, double w, double h);

    @Positive
    public boolean contains(double x, double y, double w, double h);

    @Positive
    public PathIterator getPathIterator(AffineTransform at);

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
