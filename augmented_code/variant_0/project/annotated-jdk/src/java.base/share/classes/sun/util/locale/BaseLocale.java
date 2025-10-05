/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2010, 2018, Oracle and/or its affiliates. All rights reserved.
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
package sun.util.locale;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import jdk.internal.misc.CDS;
    @Positive
import jdk.internal.vm.annotation.Stable;
    @Positive
import sun.security.action.GetPropertyAction;
    @Positive
import java.lang.ref.SoftReference;
    @Positive
import java.util.StringJoiner;

    @Positive
public final class BaseLocale {

    @Positive
    @Stable
    @Positive
    public static BaseLocale[] constantBaseLocales;

    @Positive
    public static final byte ENGLISH, FRENCH, GERMAN, ITALIAN, JAPANESE, KOREAN, CHINESE, SIMPLIFIED_CHINESE, TRADITIONAL_CHINESE, FRANCE, GERMANY, ITALY, JAPAN, KOREA, UK, US, CANADA, CANADA_FRENCH, ROOT, NUM_CONSTANTS;

    @Positive
    public static final String SEP;

    @Positive
    public static BaseLocale getInstance(String language, String script, String region, String variant);

    @Positive
    public static String convertOldISOCodes(String language);

    @Positive
    public String getLanguage();

    @Positive
    public String getScript();

    @Positive
    public String getRegion();

    @Positive
    public String getVariant();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    private static final class Key {

    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public boolean equals(Object obj);

    @Positive
        public static Key normalize(Key key);
    @Positive
    }

    @Positive
    private static class Cache extends LocaleObjectCache<Key, BaseLocale> {

    @Positive
        public Cache() {
    @Positive
        }

    @Positive
        @Override
    @Positive
        protected Key normalizeKey(Key key);

    @Positive
        @Override
    @Positive
        protected BaseLocale createObject(Key key);
    @Positive
    }
    @Positive
}
