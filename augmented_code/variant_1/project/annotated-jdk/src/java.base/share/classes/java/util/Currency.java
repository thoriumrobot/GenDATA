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
package java.util;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.BufferedInputStream;
    @Positive
import java.io.DataInputStream;
    @Positive
import java.io.File;
    @Positive
import java.io.FileReader;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.IOException;
    @Positive
import java.io.Serializable;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.text.ParseException;
    @Positive
import java.text.SimpleDateFormat;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.concurrent.ConcurrentMap;
    @Positive
import java.util.regex.Pattern;
    @Positive
import java.util.regex.Matcher;
    @Positive
import java.util.spi.CurrencyNameProvider;
    @Positive
import java.util.stream.Collectors;
    @Positive
import jdk.internal.util.StaticProperty;
    @Positive
import sun.util.locale.provider.CalendarDataUtility;
    @Positive
import sun.util.locale.provider.LocaleServiceProviderPool;
    @Positive
import sun.util.logging.PlatformLogger;

    @Positive
@AnnotatedFor({ "interning", "lock", "nullness" })
    @Positive
@SuppressWarnings("removal")
    @Positive
@UsesObjectEquals
    @Positive
public final class Currency implements Serializable {

    @Positive
    public static Currency getInstance(String currencyCode);

    @Positive
    public static Currency getInstance(Locale locale);

    @Positive
    public static Set<Currency> getAvailableCurrencies();

    @Positive
    public String getCurrencyCode(@GuardSatisfied Currency this);

    @Positive
    public String getSymbol(@GuardSatisfied Currency this);

    @Positive
    public String getSymbol(@GuardSatisfied Currency this, Locale locale);

    @Positive
    public int getDefaultFractionDigits(@GuardSatisfied Currency this);

    @Positive
    public int getNumericCode();

    @Positive
    public String getNumericCodeAsString();

    @Positive
    public String getDisplayName();

    @Positive
    public String getDisplayName(Locale locale);

    @Positive
    @SideEffectFree
    @Positive
    @Override
    @Positive
    public String toString(@GuardSatisfied Currency this);

    @Positive
    private static class CurrencyNameGetter implements LocaleServiceProviderPool.LocalizedObjectGetter<CurrencyNameProvider, String> {

    @Positive
        @Override
    @Positive
        public String getObject(CurrencyNameProvider currencyNameProvider, Locale locale, String key, Object... params);
    @Positive
    }

    @Positive
    private static class SpecialCaseEntry {
    @Positive
    }

    @Positive
    private static class OtherCurrencyEntry {
    @Positive
    }

    @Positive
    private static class CurrencyProperty {
    @Positive
    }
    @Positive
}
