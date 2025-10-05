/*
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.util.resources;

import org.checkerframework.framework.qual.AnnotatedFor;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.MissingResourceException;
import java.util.ResourceBundle;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.spi.ResourceBundleProvider;
import sun.util.locale.provider.JRELocaleProviderAdapter;
import sun.util.locale.provider.LocaleProviderAdapter;
import static sun.util.locale.provider.LocaleProviderAdapter.Type.CLDR;
import static sun.util.locale.provider.LocaleProviderAdapter.Type.JRE;
import sun.util.locale.provider.ResourceBundleBasedAdapter;

@AnnotatedFor({ "index" })
public class LocaleData {

    public LocaleData(LocaleProviderAdapter.Type type) {
    }

    public ResourceBundle getCalendarData(Locale locale);

    public OpenListResourceBundle getCurrencyNames(Locale locale);

    public OpenListResourceBundle getLocaleNames(Locale locale);

    public TimeZoneNamesBundle getTimeZoneNames(Locale locale);

    public ResourceBundle getBreakIteratorInfo(Locale locale);

    public ResourceBundle getBreakIteratorResources(Locale locale);

    public ResourceBundle getCollationData(Locale locale);

    public ResourceBundle getDateFormatData(Locale locale);

    public void setSupplementary(ParallelListResourceBundle formatData);

    public ResourceBundle getNumberFormatData(Locale locale);

    @SuppressWarnings("removal")
    public static ResourceBundle getBundle(final String baseName, final Locale locale);

    private static abstract class LocaleDataResourceBundleProvider implements ResourceBundleProvider {

        protected String toBundleName(String baseName, Locale locale);

        protected String toOtherBundleName(String baseName, String bundleName, Locale locale);
    }

    public static abstract class CommonResourceBundleProvider extends LocaleDataResourceBundleProvider {
    }

    public static abstract class SupplementaryResourceBundleProvider extends LocaleDataResourceBundleProvider {
    }

    private static class LocaleDataStrategy implements Bundles.Strategy {

        @Override
        public List<Locale> getCandidateLocales(String baseName, Locale locale);

        boolean inJavaBaseModule(String baseName, Locale locale);

        @Override
        public String toBundleName(String baseName, Locale locale);

        @Override
        public Class<? extends ResourceBundleProvider> getResourceBundleProviderType(String baseName, Locale locale);
    }

    private static class SupplementaryStrategy extends LocaleDataStrategy {

        @Override
        public List<Locale> getCandidateLocales(String baseName, Locale locale);

        @Override
        public Class<? extends ResourceBundleProvider> getResourceBundleProviderType(String baseName, Locale locale);

        @Override
        boolean inJavaBaseModule(String baseName, Locale locale);
    }
}
