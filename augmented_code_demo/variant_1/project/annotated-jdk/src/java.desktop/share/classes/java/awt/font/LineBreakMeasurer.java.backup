/*
 * Copyright (c) 1998, 2013, Oracle and/or its affiliates. All rights reserved.
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
import java.text.BreakIterator;
import java.text.CharacterIterator;
import java.text.AttributedCharacterIterator;
import java.awt.font.FontRenderContext;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public final class LineBreakMeasurer {

    public LineBreakMeasurer(AttributedCharacterIterator text, FontRenderContext frc) {
    }

    public LineBreakMeasurer(AttributedCharacterIterator text, BreakIterator breakIter, FontRenderContext frc) {
    }

    public int nextOffset(float wrappingWidth);

    public int nextOffset(float wrappingWidth, int offsetLimit, boolean requireNextWord);

    public TextLayout nextLayout(float wrappingWidth);

    public TextLayout nextLayout(float wrappingWidth, int offsetLimit, boolean requireNextWord);

    public int getPosition();

    public void setPosition(int newPosition);

    public void insertChar(AttributedCharacterIterator newParagraph, int insertPos);

    public void deleteChar(AttributedCharacterIterator newParagraph, int deletePos);
}
