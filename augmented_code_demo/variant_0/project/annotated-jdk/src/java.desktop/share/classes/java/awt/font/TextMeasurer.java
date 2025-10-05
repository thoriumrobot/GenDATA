/*
    @Positive
 * Copyright (c) 1997, 2011, Oracle and/or its affiliates. All rights reserved.
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
import java.awt.Font;
    @Positive
import java.text.AttributedCharacterIterator;
    @Positive
import java.text.AttributedCharacterIterator.Attribute;
    @Positive
import java.text.AttributedString;
    @Positive
import java.text.Bidi;
    @Positive
import java.text.BreakIterator;
    @Positive
import java.text.CharacterIterator;
    @Positive
import java.awt.font.FontRenderContext;
    @Positive
import java.util.Hashtable;
    @Positive
import java.util.Map;
    @Positive
import sun.font.AttributeValues;
    @Positive
import sun.font.BidiUtils;
    @Positive
import sun.font.TextLineComponent;
    @Positive
import sun.font.TextLabelFactory;
    @Positive
import sun.font.FontResolver;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public final class TextMeasurer implements Cloneable {

    @Positive
    public TextMeasurer(AttributedCharacterIterator text, FontRenderContext frc) {
    @Positive
    }

    @Positive
    protected Object clone();

    @Positive
    public int getLineBreakIndex(int start, float maxAdvance);

    @Positive
    public float getAdvanceBetween(int start, int limit);

    @Positive
    public TextLayout getLayout(int start, int limit);

    @Positive
    public void insertChar(AttributedCharacterIterator newParagraph, int insertPos);

    @Positive
    public void deleteChar(AttributedCharacterIterator newParagraph, int deletePos);

    @Positive
    char[] getChars();
    @Positive
}

// CFWR semantic augmentation - variant 0
