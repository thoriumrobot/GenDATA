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
public abstract class Rectangle2D extends RectangularShape {

    @Positive
    public static final int OUT_LEFT;

    @Positive
    public static final int OUT_TOP;

    @Positive
    public static final int OUT_RIGHT;

    @Positive
    public static final int OUT_BOTTOM;

    @Positive
    public static class Float extends Rectangle2D implements Serializable {

    @Positive
        public float x;

    @Positive
        public float y;

    @Positive
        public float width;

    @Positive
        public float height;

    @Positive
        public Float() {
    @Positive
        }

    @Positive
        public Float(float x, float y, float w, float h) {
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
        public boolean isEmpty();

    @Positive
        public void setRect(float x, float y, float w, float h);

    @Positive
        public void setRect(double x, double y, double w, double h);

    @Positive
        public void setRect(Rectangle2D r);

    @Positive
        public int outcode(double x, double y);

    @Positive
        public Rectangle2D getBounds2D();

    @Positive
        public Rectangle2D createIntersection(Rectangle2D r);

    @Positive
        public Rectangle2D createUnion(Rectangle2D r);

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public static class Double extends Rectangle2D implements Serializable {

    @Positive
        public double x;

    @Positive
        public double y;

    @Positive
        public double width;

    @Positive
        public double height;

    @Positive
        public Double() {
    @Positive
        }

    @Positive
        public Double(double x, double y, double w, double h) {
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
        public boolean isEmpty();

    @Positive
        public void setRect(double x, double y, double w, double h);

    @Positive
        public void setRect(Rectangle2D r);

    @Positive
        public int outcode(double x, double y);

    @Positive
        public Rectangle2D getBounds2D();

    @Positive
        public Rectangle2D createIntersection(Rectangle2D r);

    @Positive
        public Rectangle2D createUnion(Rectangle2D r);

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    protected Rectangle2D() {
    @Positive
    }

    @Positive
    public abstract void setRect(double x, double y, double w, double h);

    @Positive
    public void setRect(Rectangle2D r);

    @Positive
    public boolean intersectsLine(double x1, double y1, double x2, double y2);

    @Positive
    public boolean intersectsLine(Line2D l);

    @Positive
    public abstract int outcode(double x, double y);

    @Positive
    public int outcode(Point2D p);

    @Positive
    public void setFrame(double x, double y, double w, double h);

    @Positive
    public Rectangle2D getBounds2D();

    @Positive
    public boolean contains(double x, double y);

    @Positive
    public boolean intersects(double x, double y, double w, double h);

    @Positive
    public boolean contains(double x, double y, double w, double h);

    @Positive
    public abstract Rectangle2D createIntersection(Rectangle2D r);

    @Positive
    public static void intersect(Rectangle2D src1, Rectangle2D src2, Rectangle2D dest);

    @Positive
    public abstract Rectangle2D createUnion(Rectangle2D r);

    @Positive
    public static void union(Rectangle2D src1, Rectangle2D src2, Rectangle2D dest);

    @Positive
    public void add(double newx, double newy);

    @Positive
    public void add(Point2D pt);

    @Positive
    public void add(Rectangle2D r);

    @Positive
    public PathIterator getPathIterator(AffineTransform at);

    @Positive
    public PathIterator getPathIterator(AffineTransform at, double flatness);

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
