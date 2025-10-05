/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1998, 2013, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
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
