/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2000, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.text;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import jdk.internal.icu.text.BidiBase;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public final class Bidi {

    @Positive
    public static final int DIRECTION_LEFT_TO_RIGHT;

    @Positive
    public static final int DIRECTION_RIGHT_TO_LEFT;

    @Positive
    public static final int DIRECTION_DEFAULT_LEFT_TO_RIGHT;

    @Positive
    public static final int DIRECTION_DEFAULT_RIGHT_TO_LEFT;

    @Positive
    public Bidi(String paragraph, int flags) {
    @Positive
    }

    @Positive
    public Bidi(AttributedCharacterIterator paragraph) {
    @Positive
    }

    @Positive
    public Bidi(char[] text, int textStart, byte[] embeddings, int embStart, int paragraphLength, int flags) {
    @Positive
    }

    @Positive
    public Bidi createLineBidi(int lineStart, int lineLimit);

    @Positive
    public boolean isMixed();

    @Positive
    public boolean isLeftToRight();

    @Positive
    public boolean isRightToLeft();

    @Positive
    public int getLength();

    @Positive
    public boolean baseIsLeftToRight();

    @Positive
    public int getBaseLevel();

    @Positive
    public int getLevelAt(int offset);

    @Positive
    public int getRunCount();

    @Positive
    public int getRunLevel(int run);

    @Positive
    public int getRunStart(int run);

    @Positive
    public int getRunLimit(int run);

    @Positive
    public static boolean requiresBidi(char[] text, int start, int limit);

    @Positive
    public static void reorderVisually(byte[] levels, int levelStart, Object[] objects, int objectStart, int count);

    @Positive
    public String toString();
    @Positive
}
