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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
package java.util;

    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nullness.qual.MonotonicNonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.ObjectStreamField;
    @Positive
import java.io.Serializable;
    @Positive
import java.text.MessageFormat;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.spi.LocaleNameProvider;
    @Positive
import java.util.stream.Collectors;
    @Positive
import sun.security.action.GetPropertyAction;
    @Positive
import sun.util.locale.BaseLocale;
    @Positive
import sun.util.locale.InternalLocaleBuilder;
    @Positive
import sun.util.locale.LanguageTag;
    @Positive
import sun.util.locale.LocaleExtensions;
    @Positive
import sun.util.locale.LocaleMatcher;
    @Positive
import sun.util.locale.LocaleObjectCache;
    @Positive
import sun.util.locale.LocaleSyntaxException;
    @Positive
import sun.util.locale.LocaleUtils;
    @Positive
import sun.util.locale.ParseStatus;
    @Positive
import sun.util.locale.provider.LocaleProviderAdapter;
    @Positive
import sun.util.locale.provider.LocaleResources;
    @Positive
import sun.util.locale.provider.LocaleServiceProviderPool;
    @Positive
import sun.util.locale.provider.TimeZoneNameUtility;

    @Positive
@AnnotatedFor({ "index", "interning", "lock", "nullness" })
    @Positive
public final class Locale implements Cloneable, Serializable {

    @Positive
    public static final Locale ENGLISH;

    @Positive
    public static final Locale FRENCH;

    @Positive
    public static final Locale GERMAN;

    @Positive
    public static final Locale ITALIAN;

    @Positive
    public static final Locale JAPANESE;

    @Positive
    public static final Locale KOREAN;

    @Positive
    public static final Locale CHINESE;

    @Positive
    public static final Locale SIMPLIFIED_CHINESE;

    @Positive
    public static final Locale TRADITIONAL_CHINESE;

    @Positive
    public static final Locale FRANCE;

    @Positive
    public static final Locale GERMANY;

    @Positive
    public static final Locale ITALY;

    @Positive
    public static final Locale JAPAN;

    @Positive
    public static final Locale KOREA;

    @Positive
    public static final Locale UK;

    @Positive
    public static final Locale US;

    @Positive
    public static final Locale CANADA;

    @Positive
    public static final Locale CANADA_FRENCH;

    @Positive
    public static final Locale ROOT;

    @Positive
    public static final Locale CHINA;

    @Positive
    public static final Locale PRC;

    @Positive
    public static final Locale TAIWAN;

    @Positive
    public static final char PRIVATE_USE_EXTENSION;

    @Positive
    public static final char UNICODE_LOCALE_EXTENSION;

    @Positive
    public static enum IsoCountryCode {

    @Positive
        PART1_ALPHA2 {

    @Positive
            @Override
    @Positive
            Set<String> createCountryCodeSet();
    @Positive
        }
    @Positive
        , PART1_ALPHA3 {

    @Positive
            @Override
    @Positive
            Set<String> createCountryCodeSet();
    @Positive
        }
    @Positive
        , PART3 {

    @Positive
            @Override
    @Positive
            Set<String> createCountryCodeSet();
    @Positive
        }
    @Positive
        ;

    @Positive
        abstract Set<String> createCountryCodeSet();

    @Positive
        static Set<String> retrieveISOCountryCodes(IsoCountryCode type);
    @Positive
    }

    @Positive
    public Locale(String language, String country, String variant) {
    @Positive
    }

    @Positive
    public Locale(String language, String country) {
    @Positive
    }

    @Positive
    public Locale(String language) {
    @Positive
    }

    @Positive
    static Locale getInstance(String language, String country, String variant);

    @Positive
    static Locale getInstance(String language, String script, String country, String variant, @Nullable LocaleExtensions extensions);

    @Positive
    static Locale getInstance(BaseLocale baseloc, @Nullable LocaleExtensions extensions);

    @Positive
    private static class Cache extends LocaleObjectCache<Object, Locale> {

    @Positive
        @Override
    @Positive
        protected Locale createObject(Object key);
    @Positive
    }

    @Positive
    private static final class LocaleKey {

    @Positive
        @Override
    @Positive
        public boolean equals(Object obj);

    @Positive
        @Override
    @Positive
        public int hashCode();
    @Positive
    }

    @Positive
    public static Locale getDefault();

    @Positive
    public static Locale getDefault(Locale.Category category);

    @Positive
    public static synchronized void setDefault(Locale newLocale);

    @Positive
    public static synchronized void setDefault(Locale.Category category, Locale newLocale);

    @Positive
    public static Locale[] getAvailableLocales();

    @Positive
    public static String[] getISOCountries();

    @Positive
    public static Set<String> getISOCountries(IsoCountryCode type);

    @Positive
    public static String[] getISOLanguages();

    @Positive
    public String getLanguage();

    @Positive
    public String getScript();

    @Positive
    @Interned
    @Positive
    public String getCountry();

    @Positive
    @Interned
    @Positive
    public String getVariant();

    @Positive
    public boolean hasExtensions();

    @Positive
    public Locale stripExtensions();

    @Positive
    public String getExtension(char key);

    @Positive
    public Set<Character> getExtensionKeys();

    @Positive
    public Set<String> getUnicodeLocaleAttributes();

    @Positive
    public String getUnicodeLocaleType(String key);

    @Positive
    public Set<String> getUnicodeLocaleKeys();

    @Positive
    BaseLocale getBaseLocale();

    @Positive
    @Nullable
    @Positive
    LocaleExtensions getLocaleExtensions();

    @Positive
    @SideEffectFree
    @Positive
    @Override
    @Positive
    public final String toString();

    @Positive
    public String toLanguageTag();

    @Positive
    public static Locale forLanguageTag(String languageTag);

    @Positive
    public String getISO3Language() throws MissingResourceException;

    @Positive
    public String getISO3Country() throws MissingResourceException;

    @Positive
    public final String getDisplayLanguage();

    @Positive
    public String getDisplayLanguage(Locale inLocale);

    @Positive
    public String getDisplayScript();

    @Positive
    public String getDisplayScript(Locale inLocale);

    @Positive
    public final String getDisplayCountry();

    @Positive
    public String getDisplayCountry(Locale inLocale);

    @Positive
    public final String getDisplayVariant();

    @Positive
    public String getDisplayVariant(Locale inLocale);

    @Positive
    public final String getDisplayName();

    @Positive
    public String getDisplayName(Locale inLocale);

    @Positive
    @SideEffectFree
    @Positive
    @Override
    @Positive
    public Object clone(@GuardSatisfied Locale this);

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    private static class LocaleNameGetter implements LocaleServiceProviderPool.LocalizedObjectGetter<LocaleNameProvider, String> {

    @Positive
        @Override
    @Positive
        public String getObject(LocaleNameProvider localeNameProvider, Locale locale, String key, Object... params);
    @Positive
    }

    @Positive
    public enum Category {

    @Positive
        DISPLAY("user.language.display", "user.script.display", "user.country.display", "user.variant.display", "user.extensions.display"), FORMAT("user.language.format", "user.script.format", "user.country.format", "user.variant.format", "user.extensions.format")
    @Positive
    }

    @Positive
    public static final class Builder {

    @Positive
        public Builder() {
    @Positive
        }

    @Positive
        public Builder setLocale(Locale.@GuardSatisfied Builder this, Locale locale);

    @Positive
        public Builder setLanguageTag(Locale.@GuardSatisfied Builder this, @Nullable String languageTag);

    @Positive
        public Builder setLanguage(Locale.@GuardSatisfied Builder this, @Nullable String language);

    @Positive
        public Builder setScript(Locale.@GuardSatisfied Builder this, @Nullable String script);

    @Positive
        public Builder setRegion(Locale.@GuardSatisfied Builder this, @Nullable String region);

    @Positive
        public Builder setVariant(Locale.@GuardSatisfied Builder this, @Nullable String variant);

    @Positive
        public Builder setExtension(Locale.@GuardSatisfied Builder this, char key, @Nullable String value);

    @Positive
        public Builder setUnicodeLocaleKeyword(Locale.@GuardSatisfied Builder this, String key, @Nullable String type);

    @Positive
        public Builder addUnicodeLocaleAttribute(Locale.@GuardSatisfied Builder this, String attribute);

    @Positive
        public Builder removeUnicodeLocaleAttribute(Locale.@GuardSatisfied Builder this, String attribute);

    @Positive
        public Builder clear(Locale.@GuardSatisfied Builder this);

    @Positive
        public Builder clearExtensions(Locale.@GuardSatisfied Builder this);

    @Positive
        public Locale build();
    @Positive
    }

    @Positive
    public static enum FilteringMode {

    @Positive
        AUTOSELECT_FILTERING, EXTENDED_FILTERING, IGNORE_EXTENDED_RANGES, MAP_EXTENDED_RANGES, REJECT_EXTENDED_RANGES
    @Positive
    }

    @Positive
    public static final class LanguageRange {

    @Positive
        public static final double MAX_WEIGHT;

    @Positive
        public static final double MIN_WEIGHT;

    @Positive
        public LanguageRange(String range) {
    @Positive
        }

    @Positive
        public LanguageRange(String range, double weight) {
    @Positive
        }

    @Positive
        public String getRange();

    @Positive
        public double getWeight();

    @Positive
        public static List<LanguageRange> parse(String ranges);

    @Positive
        public static List<LanguageRange> parse(String ranges, Map<String, List<String>> map);

    @Positive
        public static List<LanguageRange> mapEquivalents(List<LanguageRange> priorityList, Map<String, List<String>> map);

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Pure
    @Positive
        @Override
    @Positive
        public boolean equals(@Nullable Object obj);

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public static List<Locale> filter(List<LanguageRange> priorityList, Collection<Locale> locales, FilteringMode mode);

    @Positive
    public static List<Locale> filter(List<LanguageRange> priorityList, Collection<Locale> locales);

    @Positive
    public static List<String> filterTags(List<LanguageRange> priorityList, Collection<String> tags, FilteringMode mode);

    @Positive
    public static List<String> filterTags(List<LanguageRange> priorityList, Collection<String> tags);

    @Positive
    @Nullable
    @Positive
    public static Locale lookup(List<LanguageRange> priorityList, Collection<Locale> locales);

    @Positive
    @Nullable
    @Positive
    public static String lookupTag(List<LanguageRange> priorityList, Collection<String> tags);
    @Positive
}
