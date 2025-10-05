/*
 * Copyright (c) 2011, 2017, Oracle and/or its affiliates. All rights reserved.
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
package sun.font;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.awt.Font;
import java.awt.font.FontRenderContext;
import java.awt.geom.AffineTransform;
import java.awt.geom.GeneralPath;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.util.ArrayList;

public final class CFont extends PhysicalFont implements FontSubstitution {

    StrikeMetrics getFontMetrics(long pScalerContext);

    float getGlyphAdvance(long pScalerContext, int glyphCode);

    void getGlyphMetrics(long pScalerContext, int glyphCode, Point2D.Float metrics);

    long getGlyphImage(long pScalerContext, int glyphCode);

    Rectangle2D.Float getGlyphOutlineBounds(long pScalerContext, int glyphCode);

    GeneralPath getGlyphOutline(long pScalerContext, int glyphCode, float x, float y);

    GeneralPath getGlyphVectorOutline(long pScalerContext, int[] glyphs, int numGlyphs, float x, float y);

    @Override
    protected byte[] getTableBytes(int tag);

    @Override
    public int getWidth();

    @Override
    public int getWeight();

    public CFont(String name) {
    }

    public CFont(String name, String inFamilyName) {
    }

    public CFont(CFont other, String logicalFamilyName) {
    }

    public CFont createItalicVariant();

    protected synchronized long getNativeFontPtr();

    protected synchronized long getPlatformNativeFontPtr();

    static native void getCascadeList(long nativeFontPtr, ArrayList<String> listOfString);

    public CompositeFont getCompositeFont2D();

    @SuppressWarnings("deprecation")
    protected synchronized void finalize();

    protected CharToGlyphMapper getMapper();

    protected FontStrike createStrike(FontStrikeDesc desc);

    public FontStrike getStrike(final Font font);

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    public int hashCode();

    public String toString();
}
