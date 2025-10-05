/*
 * Copyright (c) 1999, 2018, Oracle and/or its affiliates. All rights reserved.
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
package java.awt;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;

public final class JobAttributes implements Cloneable {

    public static final class DefaultSelectionType extends AttributeValue {

        public static final DefaultSelectionType ALL;

        public static final DefaultSelectionType RANGE;

        public static final DefaultSelectionType SELECTION;
    }

    public static final class DestinationType extends AttributeValue {

        public static final DestinationType FILE;

        public static final DestinationType PRINTER;
    }

    public static final class DialogType extends AttributeValue {

        public static final DialogType COMMON;

        public static final DialogType NATIVE;

        public static final DialogType NONE;
    }

    public static final class MultipleDocumentHandlingType extends AttributeValue {

        public static final MultipleDocumentHandlingType SEPARATE_DOCUMENTS_COLLATED_COPIES;

        public static final MultipleDocumentHandlingType SEPARATE_DOCUMENTS_UNCOLLATED_COPIES;
    }

    public static final class SidesType extends AttributeValue {

        public static final SidesType ONE_SIDED;

        public static final SidesType TWO_SIDED_LONG_EDGE;

        public static final SidesType TWO_SIDED_SHORT_EDGE;
    }

    public JobAttributes() {
    }

    public JobAttributes(JobAttributes obj) {
    }

    public JobAttributes(int copies, DefaultSelectionType defaultSelection, DestinationType destination, DialogType dialog, String fileName, int maxPage, int minPage, MultipleDocumentHandlingType multipleDocumentHandling, int[][] pageRanges, String printer, SidesType sides) {
    }

    public Object clone();

    public void set(JobAttributes obj);

    public int getCopies();

    public void setCopies(int copies);

    public void setCopiesToDefault();

    public DefaultSelectionType getDefaultSelection();

    public void setDefaultSelection(DefaultSelectionType defaultSelection);

    public DestinationType getDestination();

    public void setDestination(DestinationType destination);

    public DialogType getDialog();

    public void setDialog(DialogType dialog);

    public String getFileName();

    public void setFileName(String fileName);

    public int getFromPage();

    public void setFromPage(int fromPage);

    public int getMaxPage();

    public void setMaxPage(int maxPage);

    public int getMinPage();

    public void setMinPage(int minPage);

    public MultipleDocumentHandlingType getMultipleDocumentHandling();

    public void setMultipleDocumentHandling(MultipleDocumentHandlingType multipleDocumentHandling);

    public void setMultipleDocumentHandlingToDefault();

    public int[][] getPageRanges();

    public void setPageRanges(int[][] pageRanges);

    public String getPrinter();

    public void setPrinter(String printer);

    public SidesType getSides();

    public void setSides(SidesType sides);

    public void setSidesToDefault();

    public int getToPage();

    public void setToPage(int toPage);

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();

    public String toString();
}
