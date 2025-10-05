/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.nio.cs.ext;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import java.lang.ref.SoftReference;
    @Positive
import java.nio.charset.Charset;
    @Positive
import java.nio.charset.spi.CharsetProvider;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.TreeMap;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Map;

    @Positive
public class AbstractCharsetProvider extends CharsetProvider {

    @Positive
    protected AbstractCharsetProvider() {
    @Positive
    }

    @Positive
    protected AbstractCharsetProvider(String pkgPrefixName) {
    @Positive
    }

    @Positive
    protected void charset(String name, String className, String[] aliases);

    @Positive
    protected void deleteCharset(String name, String[] aliases);

    @Positive
    protected boolean hasCharset(String name);

    @Positive
    protected void init();

    @Positive
    public final Charset charsetForName(String charsetName);

    @Positive
    public final Iterator<Charset> charsets();

    @Positive
    public final String[] aliases(String charsetName);
    @Positive
}
