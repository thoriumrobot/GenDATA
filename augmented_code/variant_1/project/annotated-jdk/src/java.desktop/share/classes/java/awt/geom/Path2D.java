/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2006, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.Rectangle;
    @Positive
import java.awt.Shape;
    @Positive
import java.io.IOException;
    @Positive
import java.io.Serial;
    @Positive
import java.io.Serializable;
    @Positive
import java.io.StreamCorruptedException;
    @Positive
import java.util.Arrays;
    @Positive
import sun.awt.geom.Curve;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class Path2D implements Shape, Cloneable {

    @Positive
    public static final int WIND_EVEN_ODD;

    @Positive
    public static final int WIND_NON_ZERO;

    @Positive
    abstract float[] cloneCoordsFloat(AffineTransform at);

    @Positive
    abstract double[] cloneCoordsDouble(AffineTransform at);

    @Positive
    abstract void append(float x, float y);

    @Positive
    abstract void append(double x, double y);

    @Positive
    abstract Point2D getPoint(int coordindex);

    @Positive
    abstract void needRoom(boolean needMove, int newCoords);

    @Positive
    abstract int pointCrossings(double px, double py);

    @Positive
    abstract int rectCrossings(double rxmin, double rymin, double rxmax, double rymax);

    @Positive
    static byte[] expandPointTypes(byte[] oldPointTypes, int needed);

    @Positive
    public static class Float extends Path2D implements Serializable {

    @Positive
        public Float() {
    @Positive
        }

    @Positive
        public Float(int rule) {
    @Positive
        }

    @Positive
        public Float(int rule, int initialCapacity) {
    @Positive
        }

    @Positive
        public Float(Shape s) {
    @Positive
        }

    @Positive
        public Float(Shape s, AffineTransform at) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public final void trimToSize();

    @Positive
        @Override
    @Positive
        float[] cloneCoordsFloat(AffineTransform at);

    @Positive
        @Override
    @Positive
        double[] cloneCoordsDouble(AffineTransform at);

    @Positive
        void append(float x, float y);

    @Positive
        void append(double x, double y);

    @Positive
        Point2D getPoint(int coordindex);

    @Positive
        @Override
    @Positive
        void needRoom(boolean needMove, int newCoords);

    @Positive
        static float[] expandCoords(float[] oldCoords, int needed);

    @Positive
        public final synchronized void moveTo(double x, double y);

    @Positive
        public final synchronized void moveTo(float x, float y);

    @Positive
        public final synchronized void lineTo(double x, double y);

    @Positive
        public final synchronized void lineTo(float x, float y);

    @Positive
        public final synchronized void quadTo(double x1, double y1, double x2, double y2);

    @Positive
        public final synchronized void quadTo(float x1, float y1, float x2, float y2);

    @Positive
        public final synchronized void curveTo(double x1, double y1, double x2, double y2, double x3, double y3);

    @Positive
        public final synchronized void curveTo(float x1, float y1, float x2, float y2, float x3, float y3);

    @Positive
        int pointCrossings(double px, double py);

    @Positive
        int rectCrossings(double rxmin, double rymin, double rxmax, double rymax);

    @Positive
        public final void append(PathIterator pi, boolean connect);

    @Positive
        public final void transform(AffineTransform at);

    @Positive
        public final synchronized Rectangle2D getBounds2D();

    @Positive
        public final PathIterator getPathIterator(AffineTransform at);

    @Positive
        public final Object clone();

    @Positive
        static class CopyIterator extends Path2D.Iterator {

    @Positive
            public int currentSegment(float[] coords);

    @Positive
            public int currentSegment(double[] coords);
    @Positive
        }

    @Positive
        static class TxIterator extends Path2D.Iterator {

    @Positive
            public int currentSegment(float[] coords);

    @Positive
            public int currentSegment(double[] coords);
    @Positive
        }
    @Positive
    }

    @Positive
    public static class Double extends Path2D implements Serializable {

    @Positive
        public Double() {
    @Positive
        }

    @Positive
        public Double(int rule) {
    @Positive
        }

    @Positive
        public Double(int rule, int initialCapacity) {
    @Positive
        }

    @Positive
        public Double(Shape s) {
    @Positive
        }

    @Positive
        public Double(Shape s, AffineTransform at) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public final void trimToSize();

    @Positive
        @Override
    @Positive
        float[] cloneCoordsFloat(AffineTransform at);

    @Positive
        @Override
    @Positive
        double[] cloneCoordsDouble(AffineTransform at);

    @Positive
        void append(float x, float y);

    @Positive
        void append(double x, double y);

    @Positive
        Point2D getPoint(int coordindex);

    @Positive
        @Override
    @Positive
        void needRoom(boolean needMove, int newCoords);

    @Positive
        static double[] expandCoords(double[] oldCoords, int needed);

    @Positive
        public final synchronized void moveTo(double x, double y);

    @Positive
        public final synchronized void lineTo(double x, double y);

    @Positive
        public final synchronized void quadTo(double x1, double y1, double x2, double y2);

    @Positive
        public final synchronized void curveTo(double x1, double y1, double x2, double y2, double x3, double y3);

    @Positive
        int pointCrossings(double px, double py);

    @Positive
        int rectCrossings(double rxmin, double rymin, double rxmax, double rymax);

    @Positive
        public final void append(PathIterator pi, boolean connect);

    @Positive
        public final void transform(AffineTransform at);

    @Positive
        public final synchronized Rectangle2D getBounds2D();

    @Positive
        public final PathIterator getPathIterator(AffineTransform at);

    @Positive
        public final Object clone();

    @Positive
        static class CopyIterator extends Path2D.Iterator {

    @Positive
            public int currentSegment(float[] coords);

    @Positive
            public int currentSegment(double[] coords);
    @Positive
        }

    @Positive
        static class TxIterator extends Path2D.Iterator {

    @Positive
            public int currentSegment(float[] coords);

    @Positive
            public int currentSegment(double[] coords);
    @Positive
        }
    @Positive
    }

    @Positive
    public abstract void moveTo(double x, double y);

    @Positive
    public abstract void lineTo(double x, double y);

    @Positive
    public abstract void quadTo(double x1, double y1, double x2, double y2);

    @Positive
    public abstract void curveTo(double x1, double y1, double x2, double y2, double x3, double y3);

    @Positive
    public final synchronized void closePath();

    @Positive
    public final void append(Shape s, boolean connect);

    @Positive
    public abstract void append(PathIterator pi, boolean connect);

    @Positive
    public final synchronized int getWindingRule();

    @Positive
    public final void setWindingRule(int rule);

    @Positive
    public final synchronized Point2D getCurrentPoint();

    @Positive
    public final synchronized void reset();

    @Positive
    public abstract void transform(AffineTransform at);

    @Positive
    public final synchronized Shape createTransformedShape(AffineTransform at);

    @Positive
    public final Rectangle getBounds();

    @Positive
    public static boolean contains(PathIterator pi, double x, double y);

    @Positive
    public static boolean contains(PathIterator pi, Point2D p);

    @Positive
    public final boolean contains(double x, double y);

    @Positive
    public final boolean contains(Point2D p);

    @Positive
    public static boolean contains(PathIterator pi, double x, double y, double w, double h);

    @Positive
    public static boolean contains(PathIterator pi, Rectangle2D r);

    @Positive
    public final boolean contains(double x, double y, double w, double h);

    @Positive
    public final boolean contains(Rectangle2D r);

    @Positive
    public static boolean intersects(PathIterator pi, double x, double y, double w, double h);

    @Positive
    public static boolean intersects(PathIterator pi, Rectangle2D r);

    @Positive
    public final boolean intersects(double x, double y, double w, double h);

    @Positive
    public final boolean intersects(Rectangle2D r);

    @Positive
    public final PathIterator getPathIterator(AffineTransform at, double flatness);

    @Positive
    public abstract Object clone();

    @Positive
    public abstract void trimToSize();

    @Positive
    final void writeObject(java.io.ObjectOutputStream s, boolean isdbl) throws java.io.IOException;

    @Positive
    final void readObject(java.io.ObjectInputStream s, boolean storedbl) throws java.lang.ClassNotFoundException, java.io.IOException;

    @Positive
    abstract static class Iterator implements PathIterator {

    @Positive
        public int getWindingRule();

    @Positive
        public boolean isDone();

    @Positive
        public void next();
    @Positive
    }
    @Positive
}
