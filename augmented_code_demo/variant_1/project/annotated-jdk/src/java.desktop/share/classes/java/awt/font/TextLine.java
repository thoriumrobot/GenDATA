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
package java.awt.font;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.Color;
    @Positive
import java.awt.Font;
    @Positive
import java.awt.Graphics2D;
    @Positive
import java.awt.Rectangle;
    @Positive
import java.awt.Shape;
    @Positive
import java.awt.geom.AffineTransform;
    @Positive
import java.awt.geom.GeneralPath;
    @Positive
import java.awt.geom.Point2D;
    @Positive
import java.awt.geom.Rectangle2D;
    @Positive
import java.awt.im.InputMethodHighlight;
    @Positive
import java.awt.image.BufferedImage;
    @Positive
import java.text.Annotation;
    @Positive
import java.text.AttributedCharacterIterator;
    @Positive
import java.text.AttributedCharacterIterator.Attribute;
    @Positive
import java.text.Bidi;
    @Positive
import java.text.CharacterIterator;
    @Positive
import java.util.Hashtable;
    @Positive
import java.util.Map;
    @Positive
import sun.font.AttributeValues;
    @Positive
import sun.font.BidiUtils;
    @Positive
import sun.font.CodePointIterator;
    @Positive
import sun.font.CoreMetrics;
    @Positive
import sun.font.Decoration;
    @Positive
import sun.font.FontLineMetrics;
    @Positive
import sun.font.FontResolver;
    @Positive
import sun.font.GraphicComponent;
    @Positive
import sun.font.LayoutPathImpl;
    @Positive
import sun.font.LayoutPathImpl.EmptyPath;
    @Positive
import sun.font.LayoutPathImpl.SegmentPathBuilder;
    @Positive
import sun.font.TextLabelFactory;
    @Positive
import sun.font.TextLineComponent;
    @Positive
import java.awt.geom.Line2D;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
final class TextLine {

    @Positive
    static final class TextLineMetrics {

    @Positive
        public final float ascent;

    @Positive
        public final float descent;

    @Positive
        public final float leading;

    @Positive
        public final float advance;

    @Positive
        public TextLineMetrics(float ascent, float descent, float leading, float advance) {
    @Positive
        }
    @Positive
    }

    @Positive
    public TextLine(FontRenderContext frc, TextLineComponent[] components, float[] baselineOffsets, char[] chars, int charsStart, int charsLimit, int[] charLogicalOrder, byte[] charLevels, boolean isDirectionLTR) {
    @Positive
    }

    @Positive
    public Rectangle getPixelBounds(FontRenderContext frc, float x, float y);

    @Positive
    static Rectangle computePixelBounds(BufferedImage im);

    @Positive
    private abstract static class Function {

    @Positive
        abstract float computeFunction(TextLine line, int componentIndex, int indexInArray);
    @Positive
    }

    @Positive
    public int characterCount();

    @Positive
    public boolean isDirectionLTR();

    @Positive
    public TextLineMetrics getMetrics();

    @Positive
    public int visualToLogical(int visualIndex);

    @Positive
    public int logicalToVisual(int logicalIndex);

    @Positive
    public byte getCharLevel(int logicalIndex);

    @Positive
    public boolean isCharLTR(int logicalIndex);

    @Positive
    public int getCharType(int logicalIndex);

    @Positive
    public boolean isCharSpace(int logicalIndex);

    @Positive
    public boolean isCharWhitespace(int logicalIndex);

    @Positive
    public float getCharAngle(int logicalIndex);

    @Positive
    public CoreMetrics getCoreMetricsAt(int logicalIndex);

    @Positive
    public float getCharAscent(int logicalIndex);

    @Positive
    public float getCharDescent(int logicalIndex);

    @Positive
    public float getCharShift(int logicalIndex);

    @Positive
    public float getCharAdvance(int logicalIndex);

    @Positive
    public float getCharXPosition(int logicalIndex);

    @Positive
    public float getCharYPosition(int logicalIndex);

    @Positive
    public float getCharLinePosition(int logicalIndex);

    @Positive
    public float getCharLinePosition(int logicalIndex, boolean leading);

    @Positive
    public boolean caretAtOffsetIsValid(int offset);

    @Positive
    public Rectangle2D getCharBounds(int logicalIndex);

    @Positive
    public void draw(Graphics2D g2, float x, float y);

    @Positive
    public Rectangle2D getVisualBounds();

    @Positive
    public Rectangle2D getItalicBounds();

    @Positive
    public Shape getOutline(AffineTransform tx);

    @Positive
    public String toString();

    @Positive
    public static TextLine fastCreateTextLine(FontRenderContext frc, char[] chars, Font font, CoreMetrics lm, Map<? extends Attribute, ?> attributes);

    @Positive
    public static TextLineComponent[] createComponentsOnRun(int runStart, int runLimit, char[] chars, int[] charsLtoV, byte[] levels, TextLabelFactory factory, Font font, CoreMetrics cm, FontRenderContext frc, Decoration decorator, TextLineComponent[] components, int numComponents);

    @Positive
    public static TextLineComponent[] getComponents(StyledParagraph styledParagraph, char[] chars, int textStart, int textLimit, int[] charsLtoV, byte[] levels, TextLabelFactory factory);

    @Positive
    public static TextLine createLineFromText(char[] chars, StyledParagraph styledParagraph, TextLabelFactory factory, boolean isDirectionLTR, float[] baselineOffsets);

    @Positive
    public static TextLine standardCreateTextLine(FontRenderContext frc, AttributedCharacterIterator text, char[] chars, float[] baselineOffsets);

    @Positive
    static boolean advanceToFirstFont(AttributedCharacterIterator aci);

    @Positive
    static float[] getNormalizedOffsets(float[] baselineOffsets, byte baseline);

    @Positive
    static Font getFontAtCurrentPos(AttributedCharacterIterator aci);

    @Positive
    public TextLine getJustifiedLine(float justificationWidth, float justifyRatio, int justStart, int justLimit);

    @Positive
    public static float getAdvanceBetween(TextLineComponent[] components, int start, int limit);

    @Positive
    LayoutPathImpl getLayoutPath();
    @Positive
}

// CFWR semantic augmentation - variant 1
