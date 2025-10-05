/*
 * Copyright (c) 1998, 2018, Oracle and/or its affiliates. All rights reserved.
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
package sun.java2d.pipe;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.awt.Rectangle;
import java.awt.Shape;
import java.awt.geom.AffineTransform;
import java.awt.geom.Rectangle2D;
import java.awt.geom.RectangularShape;
import sun.java2d.loops.TransformHelper;
import static java.lang.Double.isNaN;

public final class Region {

    public static final Region EMPTY_REGION;

    public static final Region WHOLE_REGION;

    public static int dimAdd(int start, int dim);

    public static int clipAdd(int v, int dv);

    public static int clipRound(final double coordinate);

    public static int clipScale(final int v, final double sv);

    public static Region getInstance(Shape s, AffineTransform at);

    public static Region getInstance(Region devBounds, Shape s, AffineTransform at);

    public static Region getInstance(Region devBounds, boolean normalize, Shape s, AffineTransform at);

    static Region getInstance(final int lox, final int loy, final int hix, final int hiy, final int[] edges);

    public static Region getInstance(Rectangle r);

    public static Region getInstanceXYWH(int x, int y, int w, int h);

    public static Region getInstance(int[] box);

    public static Region getInstanceXYXY(int lox, int loy, int hix, int hiy);

    public static Region getInstance(int[] box, SpanIterator si);

    public Region getScaledRegion(final double sx, final double sy);

    public Region getTranslatedRegion(int dx, int dy);

    public Region getIntersection(Rectangle r);

    public Region getIntersectionXYWH(int x, int y, int w, int h);

    public Region getIntersection(final Rectangle2D r);

    public Region getIntersectionXYXY(double lox, double loy, double hix, double hiy);

    public Region getIntersectionXYXY(int lox, int loy, int hix, int hiy);

    public Region getIntersection(Region r);

    public Region getUnion(Region r);

    public Region getDifference(Region r);

    public Region getExclusiveOr(Region r);

    public Region getBoundsIntersection(Rectangle r);

    public Region getBoundsIntersectionXYWH(int x, int y, int w, int h);

    public Region getBoundsIntersectionXYXY(int lox, int loy, int hix, int hiy);

    public Region getBoundsIntersection(Region r);

    public int getLoX();

    public int getLoY();

    public int getHiX();

    public int getHiY();

    public int getWidth();

    public int getHeight();

    public boolean isEmpty();

    public boolean isRectangular();

    public boolean contains(int x, int y);

    public boolean isInsideXYWH(int x, int y, int w, int h);

    public boolean isInsideXYXY(int lox, int loy, int hix, int hiy);

    public boolean isInsideQuickCheck(Region r);

    public boolean intersectsQuickCheckXYXY(int lox, int loy, int hix, int hiy);

    public boolean intersectsQuickCheck(Region r);

    public boolean encompasses(Region r);

    public boolean encompassesXYWH(int x, int y, int w, int h);

    public boolean encompassesXYXY(int lox, int loy, int hix, int hiy);

    public void getBounds(int[] pathbox);

    public void clipBoxToBounds(int[] bbox);

    public RegionIterator getIterator();

    public SpanIterator getSpanIterator();

    public SpanIterator getSpanIterator(int[] bbox);

    public SpanIterator filter(SpanIterator si);

    @Override
    public String toString();

    @Override
    public int hashCode();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);
}
