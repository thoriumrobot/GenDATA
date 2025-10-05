/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.util.resources;

    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Map;
    @Positive
import java.util.MissingResourceException;
    @Positive
import java.util.ResourceBundle;
    @Positive
import java.util.Set;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.spi.ResourceBundleProvider;
    @Positive
import sun.util.locale.provider.JRELocaleProviderAdapter;
    @Positive
import sun.util.locale.provider.LocaleProviderAdapter;
    @Positive
import static sun.util.locale.provider.LocaleProviderAdapter.Type.CLDR;
    @Positive
import static sun.util.locale.provider.LocaleProviderAdapter.Type.JRE;
    @Positive
import sun.util.locale.provider.ResourceBundleBasedAdapter;

    @Positive
@AnnotatedFor({ "index" })
    @Positive
public class LocaleData {

    @Positive
    public LocaleData(LocaleProviderAdapter.Type type) {
    @Positive
    }

    @Positive
    public ResourceBundle getCalendarData(Locale locale);

    @Positive
    public OpenListResourceBundle getCurrencyNames(Locale locale);

    @Positive
    public OpenListResourceBundle getLocaleNames(Locale locale);

    @Positive
    public TimeZoneNamesBundle getTimeZoneNames(Locale locale);

    @Positive
    public ResourceBundle getBreakIteratorInfo(Locale locale);

    @Positive
    public ResourceBundle getBreakIteratorResources(Locale locale);

    @Positive
    public ResourceBundle getCollationData(Locale locale);

    @Positive
    public ResourceBundle getDateFormatData(Locale locale);

    @Positive
    public void setSupplementary(ParallelListResourceBundle formatData);

    @Positive
    public ResourceBundle getNumberFormatData(Locale locale);

    @Positive
    @SuppressWarnings("removal")
    @Positive
    public static ResourceBundle getBundle(final String baseName, final Locale locale);

    @Positive
    private static abstract class LocaleDataResourceBundleProvider implements ResourceBundleProvider {

    @Positive
        protected String toBundleName(String baseName, Locale locale);

    @Positive
        protected String toOtherBundleName(String baseName, String bundleName, Locale locale);
    @Positive
    }

    @Positive
    public static abstract class CommonResourceBundleProvider extends LocaleDataResourceBundleProvider {
    @Positive
    }

    @Positive
    public static abstract class SupplementaryResourceBundleProvider extends LocaleDataResourceBundleProvider {
    @Positive
    }

    @Positive
    private static class LocaleDataStrategy implements Bundles.Strategy {

    @Positive
        @Override
    @Positive
        public List<Locale> getCandidateLocales(String baseName, Locale locale);

    @Positive
        boolean inJavaBaseModule(String baseName, Locale locale);

    @Positive
        @Override
    @Positive
        public String toBundleName(String baseName, Locale locale);

    @Positive
        @Override
    @Positive
        public Class<? extends ResourceBundleProvider> getResourceBundleProviderType(String baseName, Locale locale);
    @Positive
    }

    @Positive
    private static class SupplementaryStrategy extends LocaleDataStrategy {

    @Positive
        @Override
    @Positive
        public List<Locale> getCandidateLocales(String baseName, Locale locale);

    @Positive
        @Override
    @Positive
        public Class<? extends ResourceBundleProvider> getResourceBundleProviderType(String baseName, Locale locale);

    @Positive
        @Override
    @Positive
        boolean inJavaBaseModule(String baseName, Locale locale);
    @Positive
    }
    @Positive
}
