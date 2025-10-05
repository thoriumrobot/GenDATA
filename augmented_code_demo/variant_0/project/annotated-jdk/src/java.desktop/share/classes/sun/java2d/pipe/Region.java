/*
    @Positive
 * Copyright (c) 1998, 2018, Oracle and/or its affiliates. All rights reserved.
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
package sun.java2d.pipe;

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
import java.awt.Rectangle;
    @Positive
import java.awt.Shape;
    @Positive
import java.awt.geom.AffineTransform;
    @Positive
import java.awt.geom.Rectangle2D;
    @Positive
import java.awt.geom.RectangularShape;
    @Positive
import sun.java2d.loops.TransformHelper;
    @Positive
import static java.lang.Double.isNaN;

    @Positive
public final class Region {

    @Positive
    public static final Region EMPTY_REGION;

    @Positive
    public static final Region WHOLE_REGION;

    @Positive
    public static int dimAdd(int start, int dim);

    @Positive
    public static int clipAdd(int v, int dv);

    @Positive
    public static int clipRound(final double coordinate);

    @Positive
    public static int clipScale(final int v, final double sv);

    @Positive
    public static Region getInstance(Shape s, AffineTransform at);

    @Positive
    public static Region getInstance(Region devBounds, Shape s, AffineTransform at);

    @Positive
    public static Region getInstance(Region devBounds, boolean normalize, Shape s, AffineTransform at);

    @Positive
    static Region getInstance(final int lox, final int loy, final int hix, final int hiy, final int[] edges);

    @Positive
    public static Region getInstance(Rectangle r);

    @Positive
    public static Region getInstanceXYWH(int x, int y, int w, int h);

    @Positive
    public static Region getInstance(int[] box);

    @Positive
    public static Region getInstanceXYXY(int lox, int loy, int hix, int hiy);

    @Positive
    public static Region getInstance(int[] box, SpanIterator si);

    @Positive
    public Region getScaledRegion(final double sx, final double sy);

    @Positive
    public Region getTranslatedRegion(int dx, int dy);

    @Positive
    public Region getIntersection(Rectangle r);

    @Positive
    public Region getIntersectionXYWH(int x, int y, int w, int h);

    @Positive
    public Region getIntersection(final Rectangle2D r);

    @Positive
    public Region getIntersectionXYXY(double lox, double loy, double hix, double hiy);

    @Positive
    public Region getIntersectionXYXY(int lox, int loy, int hix, int hiy);

    @Positive
    public Region getIntersection(Region r);

    @Positive
    public Region getUnion(Region r);

    @Positive
    public Region getDifference(Region r);

    @Positive
    public Region getExclusiveOr(Region r);

    @Positive
    public Region getBoundsIntersection(Rectangle r);

    @Positive
    public Region getBoundsIntersectionXYWH(int x, int y, int w, int h);

    @Positive
    public Region getBoundsIntersectionXYXY(int lox, int loy, int hix, int hiy);

    @Positive
    public Region getBoundsIntersection(Region r);

    @Positive
    public int getLoX();

    @Positive
    public int getLoY();

    @Positive
    public int getHiX();

    @Positive
    public int getHiY();

    @Positive
    public int getWidth();

    @Positive
    public int getHeight();

    @Positive
    public boolean isEmpty();

    @Positive
    public boolean isRectangular();

    @Positive
    public boolean contains(int x, int y);

    @Positive
    public boolean isInsideXYWH(int x, int y, int w, int h);

    @Positive
    public boolean isInsideXYXY(int lox, int loy, int hix, int hiy);

    @Positive
    public boolean isInsideQuickCheck(Region r);

    @Positive
    public boolean intersectsQuickCheckXYXY(int lox, int loy, int hix, int hiy);

    @Positive
    public boolean intersectsQuickCheck(Region r);

    @Positive
    public boolean encompasses(Region r);

    @Positive
    public boolean encompassesXYWH(int x, int y, int w, int h);

    @Positive
    public boolean encompassesXYXY(int lox, int loy, int hix, int hiy);

    @Positive
    public void getBounds(int[] pathbox);

    @Positive
    public void clipBoxToBounds(int[] bbox);

    @Positive
    public RegionIterator getIterator();

    @Positive
    public SpanIterator getSpanIterator();

    @Positive
    public SpanIterator getSpanIterator(int[] bbox);

    @Positive
    public SpanIterator filter(SpanIterator si);

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);
    @Positive
}

// CFWR semantic augmentation - variant 0
