/*
    @Positive
 * Copyright (c) 1998, 2013, Oracle and/or its affiliates. All rights reserved.
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
import java.text.BreakIterator;
    @Positive
import java.text.CharacterIterator;
    @Positive
import java.text.AttributedCharacterIterator;
    @Positive
import java.awt.font.FontRenderContext;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public final class LineBreakMeasurer {

    @Positive
    public LineBreakMeasurer(AttributedCharacterIterator text, FontRenderContext frc) {
    @Positive
    }

    @Positive
    public LineBreakMeasurer(AttributedCharacterIterator text, BreakIterator breakIter, FontRenderContext frc) {
    @Positive
    }

    @Positive
    public int nextOffset(float wrappingWidth);

    @Positive
    public int nextOffset(float wrappingWidth, int offsetLimit, boolean requireNextWord);

    @Positive
    public TextLayout nextLayout(float wrappingWidth);

    @Positive
    public TextLayout nextLayout(float wrappingWidth, int offsetLimit, boolean requireNextWord);

    @Positive
    public int getPosition();

    @Positive
    public void setPosition(int newPosition);

    @Positive
    public void insertChar(AttributedCharacterIterator newParagraph, int insertPos);

    @Positive
    public void deleteChar(AttributedCharacterIterator newParagraph, int deletePos);
    @Positive
}

// CFWR semantic augmentation - variant 0
