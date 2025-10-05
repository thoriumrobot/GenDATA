/*
    @Positive
 * Copyright (c) 1999, 2018, Oracle and/or its affiliates. All rights reserved.
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
package java.awt;

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
public final class JobAttributes implements Cloneable {

    @Positive
    public static final class DefaultSelectionType extends AttributeValue {

    @Positive
        public static final DefaultSelectionType ALL;

    @Positive
        public static final DefaultSelectionType RANGE;

    @Positive
        public static final DefaultSelectionType SELECTION;
    @Positive
    }

    @Positive
    public static final class DestinationType extends AttributeValue {

    @Positive
        public static final DestinationType FILE;

    @Positive
        public static final DestinationType PRINTER;
    @Positive
    }

    @Positive
    public static final class DialogType extends AttributeValue {

    @Positive
        public static final DialogType COMMON;

    @Positive
        public static final DialogType NATIVE;

    @Positive
        public static final DialogType NONE;
    @Positive
    }

    @Positive
    public static final class MultipleDocumentHandlingType extends AttributeValue {

    @Positive
        public static final MultipleDocumentHandlingType SEPARATE_DOCUMENTS_COLLATED_COPIES;

    @Positive
        public static final MultipleDocumentHandlingType SEPARATE_DOCUMENTS_UNCOLLATED_COPIES;
    @Positive
    }

    @Positive
    public static final class SidesType extends AttributeValue {

    @Positive
        public static final SidesType ONE_SIDED;

    @Positive
        public static final SidesType TWO_SIDED_LONG_EDGE;

    @Positive
        public static final SidesType TWO_SIDED_SHORT_EDGE;
    @Positive
    }

    @Positive
    public JobAttributes() {
    @Positive
    }

    @Positive
    public JobAttributes(JobAttributes obj) {
    @Positive
    }

    @Positive
    public JobAttributes(int copies, DefaultSelectionType defaultSelection, DestinationType destination, DialogType dialog, String fileName, int maxPage, int minPage, MultipleDocumentHandlingType multipleDocumentHandling, int[][] pageRanges, String printer, SidesType sides) {
    @Positive
    }

    @Positive
    public Object clone();

    @Positive
    public void set(JobAttributes obj);

    @Positive
    public int getCopies();

    @Positive
    public void setCopies(int copies);

    @Positive
    public void setCopiesToDefault();

    @Positive
    public DefaultSelectionType getDefaultSelection();

    @Positive
    public void setDefaultSelection(DefaultSelectionType defaultSelection);

    @Positive
    public DestinationType getDestination();

    @Positive
    public void setDestination(DestinationType destination);

    @Positive
    public DialogType getDialog();

    @Positive
    public void setDialog(DialogType dialog);

    @Positive
    public String getFileName();

    @Positive
    public void setFileName(String fileName);

    @Positive
    public int getFromPage();

    @Positive
    public void setFromPage(int fromPage);

    @Positive
    public int getMaxPage();

    @Positive
    public void setMaxPage(int maxPage);

    @Positive
    public int getMinPage();

    @Positive
    public void setMinPage(int minPage);

    @Positive
    public MultipleDocumentHandlingType getMultipleDocumentHandling();

    @Positive
    public void setMultipleDocumentHandling(MultipleDocumentHandlingType multipleDocumentHandling);

    @Positive
    public void setMultipleDocumentHandlingToDefault();

    @Positive
    public int[][] getPageRanges();

    @Positive
    public void setPageRanges(int[][] pageRanges);

    @Positive
    public String getPrinter();

    @Positive
    public void setPrinter(String printer);

    @Positive
    public SidesType getSides();

    @Positive
    public void setSides(SidesType sides);

    @Positive
    public void setSidesToDefault();

    @Positive
    public int getToPage();

    @Positive
    public void setToPage(int toPage);

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    public String toString();
    @Positive
}

// CFWR semantic augmentation - variant 0
