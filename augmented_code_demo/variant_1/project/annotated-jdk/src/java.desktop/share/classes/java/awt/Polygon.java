/*
    @Positive
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.awt;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.geom.AffineTransform;
    @Positive
import java.awt.geom.PathIterator;
    @Positive
import java.awt.geom.Point2D;
    @Positive
import java.awt.geom.Rectangle2D;
    @Positive
import java.io.Serial;
    @Positive
import java.util.Arrays;
    @Positive
import sun.awt.geom.Crossings;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class Polygon implements Shape, java.io.Serializable {

    @Positive
    public int npoints;

    @Positive
    public int[] xpoints;

    @Positive
    public int[] ypoints;

    @Positive
    protected Rectangle bounds;

    @Positive
    public Polygon() {
    @Positive
    }

    @Positive
    public Polygon(int[] xpoints, int[] ypoints, int npoints) {
    @Positive
    }

    @Positive
    public void reset();

    @Positive
    public void invalidate();

    @Positive
    public void translate(int deltaX, int deltaY);

    @Positive
    void calculateBounds(int[] xpoints, int[] ypoints, int npoints);

    @Positive
    void updateBounds(int x, int y);

    @Positive
    public void addPoint(int x, int y);

    @Positive
    public Rectangle getBounds();

    @Positive
    @Deprecated
    @Positive
    public Rectangle getBoundingBox();

    @Positive
    public boolean contains(Point p);

    @Positive
    public boolean contains(int x, int y);

    @Positive
    @Deprecated
    @Positive
    public boolean inside(int x, int y);

    @Positive
    public Rectangle2D getBounds2D();

    @Positive
    public boolean contains(double x, double y);

    @Positive
    public boolean contains(Point2D p);

    @Positive
    public boolean intersects(double x, double y, double w, double h);

    @Positive
    public boolean intersects(Rectangle2D r);

    @Positive
    public boolean contains(double x, double y, double w, double h);

    @Positive
    public boolean contains(Rectangle2D r);

    @Positive
    public PathIterator getPathIterator(AffineTransform at);

    @Positive
    public PathIterator getPathIterator(AffineTransform at, double flatness);

    @Positive
    class PolygonPathIterator implements PathIterator {

    @Positive
        public PolygonPathIterator(Polygon pg, AffineTransform at) {
    @Positive
        }

    @Positive
        public int getWindingRule();

    @Positive
        public boolean isDone();

    @Positive
        public void next();

    @Positive
        public int currentSegment(float[] coords);

    @Positive
        public int currentSegment(double[] coords);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
