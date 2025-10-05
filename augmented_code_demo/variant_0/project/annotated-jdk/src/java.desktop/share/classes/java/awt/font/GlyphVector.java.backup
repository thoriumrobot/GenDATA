/*
 * Copyright (c) 1998, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.awt.font;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.Graphics2D;
import java.awt.Font;
import java.awt.Polygon;
import java.awt.Rectangle;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.awt.geom.AffineTransform;
import java.awt.Shape;
import java.awt.font.GlyphMetrics;
import java.awt.font.GlyphJustificationInfo;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public abstract class GlyphVector implements Cloneable {

    protected GlyphVector() {
    }

    public abstract Font getFont();

    public abstract FontRenderContext getFontRenderContext();

    public abstract void performDefaultLayout();

    public abstract int getNumGlyphs();

    public abstract int getGlyphCode(int glyphIndex);

    public abstract int[] getGlyphCodes(int beginGlyphIndex, int numEntries, int[] codeReturn);

    public int getGlyphCharIndex(int glyphIndex);

    public int[] getGlyphCharIndices(int beginGlyphIndex, int numEntries, int[] codeReturn);

    public abstract Rectangle2D getLogicalBounds();

    public abstract Rectangle2D getVisualBounds();

    public Rectangle getPixelBounds(FontRenderContext renderFRC, float x, float y);

    public abstract Shape getOutline();

    public abstract Shape getOutline(float x, float y);

    public abstract Shape getGlyphOutline(int glyphIndex);

    public Shape getGlyphOutline(int glyphIndex, float x, float y);

    public abstract Point2D getGlyphPosition(int glyphIndex);

    public abstract void setGlyphPosition(int glyphIndex, Point2D newPos);

    public abstract AffineTransform getGlyphTransform(int glyphIndex);

    public abstract void setGlyphTransform(int glyphIndex, AffineTransform newTX);

    public int getLayoutFlags();

    public static final int FLAG_HAS_TRANSFORMS;

    public static final int FLAG_HAS_POSITION_ADJUSTMENTS;

    public static final int FLAG_RUN_RTL;

    public static final int FLAG_COMPLEX_GLYPHS;

    public static final int FLAG_MASK;

    public abstract float[] getGlyphPositions(int beginGlyphIndex, int numEntries, float[] positionReturn);

    public abstract Shape getGlyphLogicalBounds(int glyphIndex);

    public abstract Shape getGlyphVisualBounds(int glyphIndex);

    public Rectangle getGlyphPixelBounds(int index, FontRenderContext renderFRC, float x, float y);

    public abstract GlyphMetrics getGlyphMetrics(int glyphIndex);

    public abstract GlyphJustificationInfo getGlyphJustificationInfo(int glyphIndex);

    public abstract boolean equals(GlyphVector set);
}
