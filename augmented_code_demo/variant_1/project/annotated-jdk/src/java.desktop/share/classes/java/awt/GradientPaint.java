/*
    @Positive
 * Copyright (c) 1997, 2010, Oracle and/or its affiliates. All rights reserved.
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
import java.awt.geom.Point2D;
    @Positive
import java.awt.geom.Rectangle2D;
    @Positive
import java.awt.geom.AffineTransform;
    @Positive
import java.awt.image.ColorModel;
    @Positive
import java.beans.ConstructorProperties;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class GradientPaint implements Paint {

    @Positive
    public GradientPaint(float x1, float y1, Color color1, float x2, float y2, Color color2) {
    @Positive
    }

    @Positive
    public GradientPaint(Point2D pt1, Color color1, Point2D pt2, Color color2) {
    @Positive
    }

    @Positive
    public GradientPaint(float x1, float y1, Color color1, float x2, float y2, Color color2, boolean cyclic) {
    @Positive
    }

    @Positive
    @ConstructorProperties({ "point1", "color1", "point2", "color2", "cyclic" })
    @Positive
    public GradientPaint(Point2D pt1, Color color1, Point2D pt2, Color color2, boolean cyclic) {
    @Positive
    }

    @Positive
    public Point2D getPoint1();

    @Positive
    public Color getColor1();

    @Positive
    public Point2D getPoint2();

    @Positive
    public Color getColor2();

    @Positive
    public boolean isCyclic();

    @Positive
    public PaintContext createContext(ColorModel cm, Rectangle deviceBounds, Rectangle2D userBounds, AffineTransform xform, RenderingHints hints);

    @Positive
    public int getTransparency();
    @Positive
}

// CFWR semantic augmentation - variant 1
