/*
    @Positive
 * Copyright (c) 1998, 2020, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.Graphics2D;
    @Positive
import java.awt.Font;
    @Positive
import java.awt.Polygon;
    @Positive
import java.awt.Rectangle;
    @Positive
import java.awt.geom.Point2D;
    @Positive
import java.awt.geom.Rectangle2D;
    @Positive
import java.awt.geom.AffineTransform;
    @Positive
import java.awt.Shape;
    @Positive
import java.awt.font.GlyphMetrics;
    @Positive
import java.awt.font.GlyphJustificationInfo;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class GlyphVector implements Cloneable {

    @Positive
    protected GlyphVector() {
    @Positive
    }

    @Positive
    public abstract Font getFont();

    @Positive
    public abstract FontRenderContext getFontRenderContext();

    @Positive
    public abstract void performDefaultLayout();

    @Positive
    public abstract int getNumGlyphs();

    @Positive
    public abstract int getGlyphCode(int glyphIndex);

    @Positive
    public abstract int[] getGlyphCodes(int beginGlyphIndex, int numEntries, int[] codeReturn);

    @Positive
    public int getGlyphCharIndex(int glyphIndex);

    @Positive
    public int[] getGlyphCharIndices(int beginGlyphIndex, int numEntries, int[] codeReturn);

    @Positive
    public abstract Rectangle2D getLogicalBounds();

    @Positive
    public abstract Rectangle2D getVisualBounds();

    @Positive
    public Rectangle getPixelBounds(FontRenderContext renderFRC, float x, float y);

    @Positive
    public abstract Shape getOutline();

    @Positive
    public abstract Shape getOutline(float x, float y);

    @Positive
    public abstract Shape getGlyphOutline(int glyphIndex);

    @Positive
    public Shape getGlyphOutline(int glyphIndex, float x, float y);

    @Positive
    public abstract Point2D getGlyphPosition(int glyphIndex);

    @Positive
    public abstract void setGlyphPosition(int glyphIndex, Point2D newPos);

    @Positive
    public abstract AffineTransform getGlyphTransform(int glyphIndex);

    @Positive
    public abstract void setGlyphTransform(int glyphIndex, AffineTransform newTX);

    @Positive
    public int getLayoutFlags();

    @Positive
    public static final int FLAG_HAS_TRANSFORMS;

    @Positive
    public static final int FLAG_HAS_POSITION_ADJUSTMENTS;

    @Positive
    public static final int FLAG_RUN_RTL;

    @Positive
    public static final int FLAG_COMPLEX_GLYPHS;

    @Positive
    public static final int FLAG_MASK;

    @Positive
    public abstract float[] getGlyphPositions(int beginGlyphIndex, int numEntries, float[] positionReturn);

    @Positive
    public abstract Shape getGlyphLogicalBounds(int glyphIndex);

    @Positive
    public abstract Shape getGlyphVisualBounds(int glyphIndex);

    @Positive
    public Rectangle getGlyphPixelBounds(int index, FontRenderContext renderFRC, float x, float y);

    @Positive
    public abstract GlyphMetrics getGlyphMetrics(int glyphIndex);

    @Positive
    public abstract GlyphJustificationInfo getGlyphJustificationInfo(int glyphIndex);

    @Positive
    public abstract boolean equals(GlyphVector set);
    @Positive
}

// CFWR semantic augmentation - variant 1
