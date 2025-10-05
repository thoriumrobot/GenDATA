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
package java.util;

import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nullness.qual.MonotonicNonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.ObjectStreamField;
import java.io.Serializable;
import java.text.MessageFormat;
import java.util.concurrent.ConcurrentHashMap;
import java.util.spi.LocaleNameProvider;
import java.util.stream.Collectors;
import sun.security.action.GetPropertyAction;
import sun.util.locale.BaseLocale;
import sun.util.locale.InternalLocaleBuilder;
import sun.util.locale.LanguageTag;
import sun.util.locale.LocaleExtensions;
import sun.util.locale.LocaleMatcher;
import sun.util.locale.LocaleObjectCache;
import sun.util.locale.LocaleSyntaxException;
import sun.util.locale.LocaleUtils;
import sun.util.locale.ParseStatus;
import sun.util.locale.provider.LocaleProviderAdapter;
import sun.util.locale.provider.LocaleResources;
import sun.util.locale.provider.LocaleServiceProviderPool;
import sun.util.locale.provider.TimeZoneNameUtility;

@AnnotatedFor({ "index", "interning", "lock", "nullness" })
public final class Locale implements Cloneable, Serializable {

    public static final Locale ENGLISH;

    public static final Locale FRENCH;

    public static final Locale GERMAN;

    public static final Locale ITALIAN;

    public static final Locale JAPANESE;

    public static final Locale KOREAN;

    public static final Locale CHINESE;

    public static final Locale SIMPLIFIED_CHINESE;

    public static final Locale TRADITIONAL_CHINESE;

    public static final Locale FRANCE;

    public static final Locale GERMANY;

    public static final Locale ITALY;

    public static final Locale JAPAN;

    public static final Locale KOREA;

    public static final Locale UK;

    public static final Locale US;

    public static final Locale CANADA;

    public static final Locale CANADA_FRENCH;

    public static final Locale ROOT;

    public static final Locale CHINA;

    public static final Locale PRC;

    public static final Locale TAIWAN;

    public static final char PRIVATE_USE_EXTENSION;

    public static final char UNICODE_LOCALE_EXTENSION;

    public static enum IsoCountryCode {

        PART1_ALPHA2 {

            @Override
            Set<String> createCountryCodeSet();
        }
        , PART1_ALPHA3 {

            @Override
            Set<String> createCountryCodeSet();
        }
        , PART3 {

            @Override
            Set<String> createCountryCodeSet();
        }
        ;

        abstract Set<String> createCountryCodeSet();

        static Set<String> retrieveISOCountryCodes(IsoCountryCode type);
    }

    public Locale(String language, String country, String variant) {
    }

    public Locale(String language, String country) {
    }

    public Locale(String language) {
    }

    static Locale getInstance(String language, String country, String variant);

    static Locale getInstance(String language, String script, String country, String variant, @Nullable LocaleExtensions extensions);

    static Locale getInstance(BaseLocale baseloc, @Nullable LocaleExtensions extensions);

    private static class Cache extends LocaleObjectCache<Object, Locale> {

        @Override
        protected Locale createObject(Object key);
    }

    private static final class LocaleKey {

        @Override
        public boolean equals(Object obj);

        @Override
        public int hashCode();
    }

    public static Locale getDefault();

    public static Locale getDefault(Locale.Category category);

    public static synchronized void setDefault(Locale newLocale);

    public static synchronized void setDefault(Locale.Category category, Locale newLocale);

    public static Locale[] getAvailableLocales();

    public static String[] getISOCountries();

    public static Set<String> getISOCountries(IsoCountryCode type);

    public static String[] getISOLanguages();

    public String getLanguage();

    public String getScript();

    @Interned
    public String getCountry();

    @Interned
    public String getVariant();

    public boolean hasExtensions();

    public Locale stripExtensions();

    public String getExtension(char key);

    public Set<Character> getExtensionKeys();

    public Set<String> getUnicodeLocaleAttributes();

    public String getUnicodeLocaleType(String key);

    public Set<String> getUnicodeLocaleKeys();

    BaseLocale getBaseLocale();

    @Nullable
    LocaleExtensions getLocaleExtensions();

    @SideEffectFree
    @Override
    public final String toString();

    public String toLanguageTag();

    public static Locale forLanguageTag(String languageTag);

    public String getISO3Language() throws MissingResourceException;

    public String getISO3Country() throws MissingResourceException;

    public final String getDisplayLanguage();

    public String getDisplayLanguage(Locale inLocale);

    public String getDisplayScript();

    public String getDisplayScript(Locale inLocale);

    public final String getDisplayCountry();

    public String getDisplayCountry(Locale inLocale);

    public final String getDisplayVariant();

    public String getDisplayVariant(Locale inLocale);

    public final String getDisplayName();

    public String getDisplayName(Locale inLocale);

    @SideEffectFree
    @Override
    public Object clone(@GuardSatisfied Locale this);

    @Pure
    @Override
    public int hashCode();

    @Pure
    @Override
    public boolean equals(@Nullable Object obj);

    private static class LocaleNameGetter implements LocaleServiceProviderPool.LocalizedObjectGetter<LocaleNameProvider, String> {

        @Override
        public String getObject(LocaleNameProvider localeNameProvider, Locale locale, String key, Object... params);
    }

    public enum Category {

        DISPLAY("user.language.display", "user.script.display", "user.country.display", "user.variant.display", "user.extensions.display"), FORMAT("user.language.format", "user.script.format", "user.country.format", "user.variant.format", "user.extensions.format")
    }

    public static final class Builder {

        public Builder() {
        }

        public Builder setLocale(Locale.@GuardSatisfied Builder this, Locale locale);

        public Builder setLanguageTag(Locale.@GuardSatisfied Builder this, @Nullable String languageTag);

        public Builder setLanguage(Locale.@GuardSatisfied Builder this, @Nullable String language);

        public Builder setScript(Locale.@GuardSatisfied Builder this, @Nullable String script);

        public Builder setRegion(Locale.@GuardSatisfied Builder this, @Nullable String region);

        public Builder setVariant(Locale.@GuardSatisfied Builder this, @Nullable String variant);

        public Builder setExtension(Locale.@GuardSatisfied Builder this, char key, @Nullable String value);

        public Builder setUnicodeLocaleKeyword(Locale.@GuardSatisfied Builder this, String key, @Nullable String type);

        public Builder addUnicodeLocaleAttribute(Locale.@GuardSatisfied Builder this, String attribute);

        public Builder removeUnicodeLocaleAttribute(Locale.@GuardSatisfied Builder this, String attribute);

        public Builder clear(Locale.@GuardSatisfied Builder this);

        public Builder clearExtensions(Locale.@GuardSatisfied Builder this);

        public Locale build();
    }

    public static enum FilteringMode {

        AUTOSELECT_FILTERING, EXTENDED_FILTERING, IGNORE_EXTENDED_RANGES, MAP_EXTENDED_RANGES, REJECT_EXTENDED_RANGES
    }

    public static final class LanguageRange {

        public static final double MAX_WEIGHT;

        public static final double MIN_WEIGHT;

        public LanguageRange(String range) {
        }

        public LanguageRange(String range, double weight) {
        }

        public String getRange();

        public double getWeight();

        public static List<LanguageRange> parse(String ranges);

        public static List<LanguageRange> parse(String ranges, Map<String, List<String>> map);

        public static List<LanguageRange> mapEquivalents(List<LanguageRange> priorityList, Map<String, List<String>> map);

        @Override
        public int hashCode();

        @Pure
        @Override
        public boolean equals(@Nullable Object obj);

        @Override
        public String toString();
    }

    public static List<Locale> filter(List<LanguageRange> priorityList, Collection<Locale> locales, FilteringMode mode);

    public static List<Locale> filter(List<LanguageRange> priorityList, Collection<Locale> locales);

    public static List<String> filterTags(List<LanguageRange> priorityList, Collection<String> tags, FilteringMode mode);

    public static List<String> filterTags(List<LanguageRange> priorityList, Collection<String> tags);

    @Nullable
    public static Locale lookup(List<LanguageRange> priorityList, Collection<Locale> locales);

    @Nullable
    public static String lookupTag(List<LanguageRange> priorityList, Collection<String> tags);
}
