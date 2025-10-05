/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.print;

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
import java.awt.GraphicsEnvironment;
    @Positive
import java.awt.Toolkit;
    @Positive
import java.awt.Window;
    @Positive
import java.awt.print.PrinterJob;
    @Positive
import java.io.File;
    @Positive
import java.net.URI;
    @Positive
import java.net.URISyntaxException;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.HashMap;
    @Positive
import javax.print.DocFlavor;
    @Positive
import javax.print.DocPrintJob;
    @Positive
import javax.print.PrintService;
    @Positive
import javax.print.ServiceUIFactory;
    @Positive
import javax.print.attribute.Attribute;
    @Positive
import javax.print.attribute.AttributeSet;
    @Positive
import javax.print.attribute.AttributeSetUtilities;
    @Positive
import javax.print.attribute.EnumSyntax;
    @Positive
import javax.print.attribute.HashAttributeSet;
    @Positive
import javax.print.attribute.PrintRequestAttributeSet;
    @Positive
import javax.print.attribute.PrintServiceAttribute;
    @Positive
import javax.print.attribute.PrintServiceAttributeSet;
    @Positive
import javax.print.attribute.HashPrintServiceAttributeSet;
    @Positive
import javax.print.attribute.standard.PrinterName;
    @Positive
import javax.print.attribute.standard.PrinterIsAcceptingJobs;
    @Positive
import javax.print.attribute.standard.QueuedJobCount;
    @Positive
import javax.print.attribute.standard.JobName;
    @Positive
import javax.print.attribute.standard.RequestingUserName;
    @Positive
import javax.print.attribute.standard.Chromaticity;
    @Positive
import javax.print.attribute.standard.Copies;
    @Positive
import javax.print.attribute.standard.CopiesSupported;
    @Positive
import javax.print.attribute.standard.Destination;
    @Positive
import javax.print.attribute.standard.DialogOwner;
    @Positive
import javax.print.attribute.standard.DialogTypeSelection;
    @Positive
import javax.print.attribute.standard.Fidelity;
    @Positive
import javax.print.attribute.standard.Media;
    @Positive
import javax.print.attribute.standard.MediaSizeName;
    @Positive
import javax.print.attribute.standard.MediaSize;
    @Positive
import javax.print.attribute.standard.MediaTray;
    @Positive
import javax.print.attribute.standard.MediaPrintableArea;
    @Positive
import javax.print.attribute.standard.OrientationRequested;
    @Positive
import javax.print.attribute.standard.PageRanges;
    @Positive
import javax.print.attribute.standard.PrinterState;
    @Positive
import javax.print.attribute.standard.PrinterStateReason;
    @Positive
import javax.print.attribute.standard.PrinterStateReasons;
    @Positive
import javax.print.attribute.standard.Severity;
    @Positive
import javax.print.attribute.standard.Sides;
    @Positive
import javax.print.attribute.standard.ColorSupported;
    @Positive
import javax.print.attribute.standard.PrintQuality;
    @Positive
import javax.print.attribute.standard.PrinterResolution;
    @Positive
import javax.print.attribute.standard.SheetCollate;
    @Positive
import javax.print.event.PrintServiceAttributeListener;
    @Positive
import sun.awt.windows.WPrinterJob;

    @Positive
public class Win32PrintService implements PrintService, AttributeUpdater, SunPrinterJobService {

    @Positive
    public static MediaSize[] predefMedia;

    @Positive
    public static final MediaSizeName[] dmPaperToPrintService;

    @Positive
    public void invalidateService();

    @Positive
    public String getName();

    @Positive
    public int findPaperID(MediaSizeName msn);

    @Positive
    public int findTrayID(MediaTray tray);

    @Positive
    public MediaTray findMediaTray(int dmBin);

    @Positive
    public MediaSizeName findWin32Media(int dmIndex);

    @Positive
    public MediaSizeName findMatchingMediaSizeNameMM(float w, float h);

    @Positive
    public DocPrintJob createPrintJob();

    @Positive
    public PrintServiceAttributeSet getUpdatedAttributes();

    @Positive
    public void wakeNotifier();

    @Positive
    public void addPrintServiceAttributeListener(PrintServiceAttributeListener listener);

    @Positive
    public void removePrintServiceAttributeListener(PrintServiceAttributeListener listener);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public <T extends PrintServiceAttribute> T getAttribute(Class<T> category);

    @Positive
    public PrintServiceAttributeSet getAttributes();

    @Positive
    public DocFlavor[] getSupportedDocFlavors();

    @Positive
    public boolean isDocFlavorSupported(DocFlavor flavor);

    @Positive
    public Class<?>[] getSupportedAttributeCategories();

    @Positive
    public boolean isAttributeCategorySupported(Class<? extends Attribute> category);

    @Positive
    public Object getDefaultAttributeValue(Class<? extends Attribute> category);

    @Positive
    public Object getSupportedAttributeValues(Class<? extends Attribute> category, DocFlavor flavor, AttributeSet attributes);

    @Positive
    public boolean isAttributeValueSupported(Attribute attr, DocFlavor flavor, AttributeSet attributes);

    @Positive
    public AttributeSet getUnsupportedAttributes(DocFlavor flavor, AttributeSet attributes);

    @Positive
    private static class Win32DocumentPropertiesUI extends DocumentPropertiesUI {

    @Positive
        public PrintRequestAttributeSet showDocumentProperties(PrinterJob job, Window owner, PrintService service, PrintRequestAttributeSet aset);
    @Positive
    }

    @Positive
    private static class Win32ServiceUIFactory extends ServiceUIFactory {

    @Positive
        public Object getUI(int role, String ui);

    @Positive
        public String[] getUIClassNamesForRole(int role);
    @Positive
    }

    @Positive
    public synchronized ServiceUIFactory getServiceUIFactory();

    @Positive
    public String toString();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    public boolean usesClass(Class<?> c);
    @Positive
}

    @Positive
@SuppressWarnings("serial")
    @Positive
class Win32MediaSize extends MediaSizeName {

    @Positive
    public static synchronized Win32MediaSize findMediaName(String name);

    @Positive
    public static MediaSize[] getPredefMedia();

    @Positive
    public Win32MediaSize(String name, int dmPaper) {
    @Positive
    }

    @Positive
    int getDMPaper();

    @Positive
    protected String[] getStringTable();

    @Positive
    protected EnumSyntax[] getEnumValueTable();
    @Positive
}
