/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
import java.awt.font.FontRenderContext;
    @Positive
import java.awt.font.LineMetrics;
    @Positive
import java.awt.geom.Rectangle2D;
    @Positive
import java.io.Serial;
    @Positive
import java.text.CharacterIterator;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class FontMetrics implements java.io.Serializable {

    @Positive
    protected Font font;

    @Positive
    protected FontMetrics(Font font) {
    @Positive
    }

    @Positive
    public Font getFont();

    @Positive
    public FontRenderContext getFontRenderContext();

    @Positive
    public int getLeading();

    @Positive
    public int getAscent();

    @Positive
    public int getDescent();

    @Positive
    public int getHeight();

    @Positive
    public int getMaxAscent();

    @Positive
    public int getMaxDescent();

    @Positive
    @Deprecated
    @Positive
    public int getMaxDecent();

    @Positive
    public int getMaxAdvance();

    @Positive
    public int charWidth(int codePoint);

    @Positive
    public int charWidth(char ch);

    @Positive
    public int stringWidth(String str);

    @Positive
    public int charsWidth(char[] data, int off, int len);

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    public int bytesWidth(byte[] data, int off, int len);

    @Positive
    public int[] getWidths();

    @Positive
    public boolean hasUniformLineMetrics();

    @Positive
    public LineMetrics getLineMetrics(String str, Graphics context);

    @Positive
    public LineMetrics getLineMetrics(String str, int beginIndex, int limit, Graphics context);

    @Positive
    public LineMetrics getLineMetrics(char[] chars, int beginIndex, int limit, Graphics context);

    @Positive
    public LineMetrics getLineMetrics(CharacterIterator ci, int beginIndex, int limit, Graphics context);

    @Positive
    public Rectangle2D getStringBounds(String str, Graphics context);

    @Positive
    public Rectangle2D getStringBounds(String str, int beginIndex, int limit, Graphics context);

    @Positive
    public Rectangle2D getStringBounds(char[] chars, int beginIndex, int limit, Graphics context);

    @Positive
    public Rectangle2D getStringBounds(CharacterIterator ci, int beginIndex, int limit, Graphics context);

    @Positive
    public Rectangle2D getMaxCharBounds(Graphics context);

    @Positive
    public String toString();
    @Positive
}
