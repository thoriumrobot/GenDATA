/*
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.awt;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.geom.AffineTransform;
import java.awt.geom.PathIterator;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.io.Serial;
import java.util.Arrays;
import sun.awt.geom.Crossings;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class Polygon implements Shape, java.io.Serializable {

    public int npoints;

    public int[] xpoints;

    public int[] ypoints;

    protected Rectangle bounds;

    public Polygon() {
    }

    public Polygon(int[] xpoints, int[] ypoints, int npoints) {
    }

    public void reset();

    public void invalidate();

    public void translate(int deltaX, int deltaY);

    void calculateBounds(int[] xpoints, int[] ypoints, int npoints);

    void updateBounds(int x, int y);

    public void addPoint(int x, int y);

    public Rectangle getBounds();

    @Deprecated
    public Rectangle getBoundingBox();

    public boolean contains(Point p);

    public boolean contains(int x, int y);

    @Deprecated
    public boolean inside(int x, int y);

    public Rectangle2D getBounds2D();

    public boolean contains(double x, double y);

    public boolean contains(Point2D p);

    public boolean intersects(double x, double y, double w, double h);

    public boolean intersects(Rectangle2D r);

    public boolean contains(double x, double y, double w, double h);

    public boolean contains(Rectangle2D r);

    public PathIterator getPathIterator(AffineTransform at);

    public PathIterator getPathIterator(AffineTransform at, double flatness);

    class PolygonPathIterator implements PathIterator {

        public PolygonPathIterator(Polygon pg, AffineTransform at) {
        }

        public int getWindingRule();

        public boolean isDone();

        public void next();

        public int currentSegment(float[] coords);

        public int currentSegment(double[] coords);
    }
}
