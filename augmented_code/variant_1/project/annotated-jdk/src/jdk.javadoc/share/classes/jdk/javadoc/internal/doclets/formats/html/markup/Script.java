/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2003, 2019, Oracle and/or its affiliates. All rights reserved.
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
package jdk.javadoc.internal.doclets.formats.html.markup;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.io.IOException;
    @Positive
import java.io.Writer;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.Content;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.DocletConstants;

    @Positive
public class Script {

    @Positive
    public Script() {
    @Positive
    }

    @Positive
    public Script(String code) {
    @Positive
    }

    @Positive
    public Script append(CharSequence code);

    @Positive
    public Script appendStringLiteral(CharSequence text);

    @Positive
    public Script appendStringLiteral(CharSequence text, char quoteChar);

    @Positive
    public Content asContent();

    @Positive
    public static String stringLiteral(CharSequence s);

    @Positive
    public static String stringLiteral(CharSequence s, char quoteChar);

    @Positive
    private static class ScriptContent extends Content {

    @Positive
        @Override
    @Positive
        public ScriptContent add(CharSequence code);

    @Positive
        @Override
    @Positive
        public boolean write(Writer writer, boolean atNewline) throws IOException;

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        public boolean isEmpty();
    @Positive
    }
    @Positive
}
