/*
    @Positive
 * Copyright (c) 1998, 2006, Oracle and/or its affiliates. All rights reserved.
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
package java.awt.font;

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
import java.awt.Graphics;
    @Positive
import java.awt.Rectangle;
    @Positive
import java.awt.Graphics2D;
    @Positive
import java.awt.Shape;
    @Positive
import java.awt.geom.AffineTransform;
    @Positive
import java.awt.geom.Rectangle2D;

    @Positive
public final class ShapeGraphicAttribute extends GraphicAttribute {

    @Positive
    public static final boolean STROKE;

    @Positive
    public static final boolean FILL;

    @Positive
    public ShapeGraphicAttribute(Shape shape, int alignment, boolean stroke) {
    @Positive
    }

    @Positive
    public float getAscent();

    @Positive
    public float getDescent();

    @Positive
    public float getAdvance();

    @Positive
    public void draw(Graphics2D graphics, float x, float y);

    @Positive
    public Rectangle2D getBounds();

    @Positive
    public Shape getOutline(AffineTransform tx);

    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object rhs);

    @Positive
    public boolean equals(ShapeGraphicAttribute rhs);
    @Positive
}

// CFWR semantic augmentation - variant 1
